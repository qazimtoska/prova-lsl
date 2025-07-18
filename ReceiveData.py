# Pipeline di acquisizione, inferenza e controllo

from pylsl import StreamInlet, resolve_stream

import time
import sys
import numpy as np
from enum import Enum
import json
import pickle
import os
import re

import mne
import tflite_runtime.interpreter as tflite

import threading
from collections import Counter, deque

import paho.mqtt.client as mqtt

from periphery import Serial

uart1 = Serial("/dev/ttymxc0", 115200)

# Specifica come inviare il comando: seriale (s) o MQTT (m). Default: s
send_method = None

# Parametri di connessione al broker MQTT
broker_address = "149.132.176.75"  
broker_port = 1883
username = "Wheelchair"
password = "BCI-Wh33lch41r"

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
topic = "/ISLab/BCI/Wheelchair/control"
qos_level = 0
keep_alive_interval = 60

# Comandi da inviare all'ESP32 tramite seriale
FORWARD = "AT+FORWARD={}\n"
BACKWARD = "AT+BACKWARD={}\n"
LEFT = "AT+LEFT={}\n"
RIGHT = "AT+RIGHT={}\n"
STOP = "AT+STOP\n"

# Schema del comando da inviare all'ESP32 tramite MQTT
mqtt_command = {
    "command": None,
    "value": 100
}

# Modello TFLite
try:
    edgetpu_delegate = tflite.load_delegate('libedgetpu.so.1')
    print("Verrà utilizzata la TPU per la fase di inferenza.")
    interpreter = tflite.Interpreter(
        model_path="model_edgetpu.tflite",
        experimental_delegates=[edgetpu_delegate]
    )
except Exception as e:
    print("Edge TPU non disponibile, fallback su CPU")
    interpreter = tflite.Interpreter(model_path="model_edgetpu.tflite")

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Oggetto Condition() che mi serve per regolare le scritture/letture di predictions tramite 
# acquisizione e rilascio di un Lock
condition = threading.Condition()

# Lista che contiene in ciascuna posizione una coda (scorrevole) di lunghezza max = 5.
# Ciascuna coda rappresenta una label e ne contiene i pesi più recenti, prodotti dai
# thread di inferenza.
weights_matrix = [deque(maxlen=5), deque(maxlen=5), deque(maxlen=5), deque(maxlen=5), deque(maxlen=5)]

# Ultima predizione
last_prediction = None

class Prediction(Enum):
    STOP = 0
    LEFT = 1
    RIGHT = 2
    FORWARD = 3
    BACKWARD = 4

def majority_voting():
    """
    Il thread majority_voting è costantemente in uno stato di wait, ad eccezione di quando viene 
    risvegliato da un thread inference (dopo aver prodotto una label). In quest'ultima circostanza
    fa voto maggioritario fra le label presenti in predictions
    """
    global last_prediction

    def send_command(prediction):
        if send_method == 's': # Scrittura su seriale
            if prediction == Prediction.STOP.value:
                command = STOP
            elif prediction == Prediction.LEFT.value:
                command = LEFT.format(100)
            elif prediction == Prediction.RIGHT.value:
                command = RIGHT.format(100)
            elif prediction == Prediction.FORWARD.value:
                command = FORWARD.format(100)
            else:
                command = BACKWARD.format(100)
            uart1.write(command.encode('utf-8'))
        else: # Invio comando con MQTT
            if prediction == Prediction.STOP.value:
                mqtt_command["command"] = "STOP"
            elif prediction == Prediction.LEFT.value:
                mqtt_command["command"] = "LEFT"
            elif prediction == Prediction.RIGHT.value:
                mqtt_command["command"] = "RIGHT"
            elif prediction == Prediction.FORWARD.value:
                mqtt_command["command"] = "FORWARD"
            else:
                mqtt_command["command"] = "BACKWARD"
            result = client.publish(topic, json.dumps(mqtt_command))
            result.wait_for_publish()

    while True:
        with condition:
            condition.wait()

            # weight_sum = array che contiene in ciascuna posizione la somma
            # dei pesi di ciascuna coda
            weight_sum = [sum(weight_queue) for weight_queue in weights_matrix]

            max_weight_sum = max(weight_sum)
            max_indices = [index for index, value in enumerate(weight_sum) if value == max_weight_sum]

            if len(max_indices) > 1: # situazione di pareggio
                if not last_prediction:
                    # Ordine di priorità nel caso in cui questa sia la prima predizione
                    if Prediction.STOP.value in max_indices:
                        prediction = Prediction.STOP.value
                    elif Prediction.FORWARD.value in max_indices:
                        prediction = Prediction.FORWARD.value
                    elif Prediction.RIGHT.value in max_indices:
                        prediction = Prediction.RIGHT.value
                    elif Prediction.LEFT.value in max_indices:
                        prediction = Prediction.LEFT.value
                    else:
                        prediction = Prediction.BACKWARD.value
                else:
                    # Se c'è un pareggio ma è stata già effettuata almeno una predizione,
                    # allora predizione attuale = ultima predizione fatta
                    prediction = last_prediction
            else:
                prediction = max_indices[0]

            print("Prediction", prediction)
            send_command(prediction)

            last_prediction = prediction

def inference(data_matrix, info):
    # Il modello deve selezionare i 4.5 secondi più recenti dalla finestra di 5 secondi.
    # Ecco perché taglio la matrice, facendola partire dalla colonna 79 (non 80 poiché 
    # mne vuole un segmento da 4.5 secondi + 1 ulteriore campione => 721 campioni)
    data_matrix = data_matrix[:, 79:]

    # A partire da data_matrix costruisco un oggetto dal quale poter creare successivamente 
    # la singola epoca
    segmentoRaw = mne.io.RawArray(data_matrix, info)
    segmentoRaw = segmentoRaw.pick("eeg", exclude='bads') 

    epochs = mne.Epochs(
        segmentoRaw,
        # La colonna 0 specifica da quale sample parte il segmento, la 2 indica una predizione falsa
        # per questo segmento (necessario). La 1 non è importante
        np.array([[0, 0, 1]]),
        tmin=0.0,
        tmax=4.5,
        baseline=None,
        preload=True,
        verbose=False
    )

    data = epochs.get_data() * 1e6

    input_data = np.array(data, dtype=np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])  # shape: (1, 3)

    weights = output_data[0].tolist()

    with condition:
        # aggiorno la matrice di pesi con i nuovi pesi
        for i in range(3):
            weights_matrix[i].append(weights[i])
        weights_matrix[3].append(0)
        weights_matrix[4].append(0)

        condition.notify()

def save_recording(file_matrix):
    # N.B. Essendo che il programma viene eseguito con 'sudo', per rimuovere la cartella e le registrazioni occorre eseguire sudo rm -rf recordings 
    folder = "recordings"
    os.makedirs(folder, exist_ok=True)

    existing_files = os.listdir(folder)
    pattern = re.compile(r"data_(\d+)\.pkl")
    numbers = [int(pattern.match(f).group(1)) for f in existing_files if pattern.match(f)]

    next_number = max(numbers) + 1 if numbers else 0
    filename = f"data_{next_number:02d}.pkl"
    file_path = os.path.join(folder, filename)

    with open(file_path, "wb") as file:
        pickle.dump(file_matrix, file)

    print(f"Registrazione salvata in {file_path}")

def main():
    global send_method

    send_method = sys.argv[1] if len(sys.argv) > 1 else 's'
    if send_method not in ('s', 'm'):
        print("Invalid argument. Defaulting to 's' = writing to serial.")
        send_method = 's'

    if send_method == 'm':
        # Connessione al broker MQTT
        client.username_pw_set(username, password)
        client.connect(broker_address, broker_port, keepalive=keep_alive_interval)
        client.loop_start()
        client.default_qos = qos_level

    # Caricamento file edf solo per produrre l'oggetto di informazioni necessario a ricostruire
    # i vari segmenti
    edf_files = mne.datasets.eegbci.load_data(1, [4])
    raw = mne.io.read_raw_edf(edf_files[0], preload=True, stim_channel='auto', verbose=False)
    info = raw.info

    print("Rilevazione stream EEG...")
    streams = resolve_stream('type', 'EEG')

    """
    data_matrix è una matrice (n_channels X n_samples), dove n_samples indica la lunghezza 
    della finestra. Vogliamo che la finestra sia costantemente di 5 secondi, ma anche che sia
    scorrevole, dunque che i vecchi dati lascino posto ai nuovi, che verranno inseriti alla fine.

    data_matrix è paragonabile a raw_matrix nel file SendData.py, ma differisce nel numero di colonne.
    Difatti data_matrix contiene i dati di un singolo segmento temporale, di cui faremo inferenza, non 
    dell'intera registrazione

    Inizialmente non ci sono dati per ciascuno dei 64 canali
    """
    data_matrix = np.empty((64, 0))

    # Matrice che contiene l'intero flusso di registrazione, il quale sarà salvato nel file data.pkl
    file_matrix = np.empty((64, 0), dtype=np.float32)

    # Consente di leggere dallo stream
    inlet = StreamInlet(streams[0])

    # Lista che mi consente di fare il join dei thread inference
    threads = []
    
    votingThread = threading.Thread(target=majority_voting)
    votingThread.start()

    # Mediante questo ciclo raccolgo i primi 5 secondi di segnale
    while True:
        sample, _ = inlet.pull_sample() # sample = campione per ciascuno dei 64 canali
        sample_column = np.array(sample).reshape(64, 1)

        # aggiungo la colonna ottenuta in fondo alla matrice data_matrix
        data_matrix = np.hstack((data_matrix, sample_column))
        file_matrix = np.hstack((file_matrix, sample_column))

        if (len(data_matrix[0]) % 800 == 0): # 800 => 800 * 0.00625 = 5 secondi
            break

    try:
        # Mediante questo ciclo raccolgo costantemente nuovi campioni e ogni 0.3 secondi faccio inferenza
        while True:
            x = threading.Thread(target=inference, args=(data_matrix,info,))
            threads.append(x)
            x.start()

            # 48 samples corrispondono a 0.3 secondi di segnale
            for _ in range(48):
                sample, _ = inlet.pull_sample()
                sample_column = np.array(sample).reshape(64, 1)
                data_matrix = np.hstack((data_matrix, sample_column))
                file_matrix = np.hstack((file_matrix, sample_column))

                # Ogni campione ricevuto viene incolonnato alla fine di data_matrix, e per mantenere
                # la stessa dimensione del segmento (5 secondi) la prima colonna (0) viene rimossa.
                # Dunque data_matrix è una matrice scorrevole
                data_matrix = data_matrix[:, 1:]

    except KeyboardInterrupt:
        print("\nInterruzione rilevata. Attendo la fine dei thread...")
    
    # Se il programma viene interrotto dall'utente si attende che tutte le predizioni siano state effettuate
    for _, thread in enumerate(threads):
        thread.join()
    
    save_recording(file_matrix)
    
    print("È possibile chiudere il programma.")

if __name__ == '__main__':
    main()
