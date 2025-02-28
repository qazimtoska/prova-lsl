"""Example program to show how to read a multi-channel time series from LSL."""

from pylsl import StreamInlet, resolve_stream

import time
import numpy as np

import mne
import torch
from torch.utils.data import DataLoader
from models import EEGSegment, MultiStream1DCNN

import threading

import paho.mqtt.client as mqtt

from periphery import Serial

uart1 = Serial("/dev/ttymxc0", 115200)

# Comandi da inviare all'ESP32
FORWARD = "AT+FORWARD={}\n"
BACK = "AT+BACK={}\n"
LEFT = "AT+LEFT={}\n"
RIGHT = "AT+RIGHT={}\n"
STOP = "AT+STOP={}\n"

# Parametri di connessione al broker MQTT
broker_address = "broker.emqx.io"  
broker_port = 1883  
username = ""  
password = ""

client = mqtt.Client()
topic = "edf_files"
qos_level = 0
keep_alive_interval = 60

# Esempio: usiamo il modello allenato del fold 1
in_channels = 64
device = "cpu"
model_inference = MultiStream1DCNN(in_channels=in_channels, n_classes=3).to(device)
model_inference.load_state_dict(torch.load("model_fold_1.pth", map_location=device))
model_inference.eval()

def inference(receiving_matrix, info):
    def send_command(prediction):
        if prediction == 0:
            command = STOP.format(127)
        elif prediction == 1:
            command = LEFT.format(127)
        elif prediction == 2:
            command = RIGHT.format(127)
        elif prediction == 3:
            command = FORWARD.format(127)
        else:
            command = BACK.format(127)

        # Scrittura comando tramite seriale
        uart1.write(command.encode('utf-8'))
        
        # Invio comando con MQTT
        result = client.publish(topic, command)
        result.wait_for_publish()
    
    receiving_matrix = np.array(receiving_matrix)

    segmentoRaw = mne.io.RawArray(receiving_matrix, info)
    segmentoRaw = segmentoRaw.pick("eeg", exclude='bads') 

    epochs = mne.Epochs(
        segmentoRaw,
        np.array([[0, 0, 1]]),
        tmin=0.0,
        tmax=4.5,
        baseline=None,
        preload=True,
        verbose=False
    )

    data = epochs.get_data() * 1e6

    data_tensor = torch.tensor(data, dtype=torch.float32)
    example_segment = EEGSegment(data_tensor)
    example_loader = DataLoader(example_segment, batch_size=1, shuffle=False)

    with torch.no_grad():
        for xb, yb in example_loader:
            xb = xb.to(device)
            logits = model_inference(xb)
            _, preds = logits.max(dim=1)
            print("Predizione:", preds.numpy()[0])
            send_command(preds.numpy()[0])

def main():
    # Caricamento file edf
    edf_files = mne.datasets.eegbci.load_data(1, [4])
    raw = mne.io.read_raw_edf(edf_files[0], preload=True, stim_channel='auto', verbose=False)
    info = raw.info

    # Connessione al broker MQTT
    client.connect(broker_address, broker_port, keepalive=keep_alive_interval)
    client.loop_start()
    client.default_qos = qos_level

    # first resolve an EEG stream on the lab network
    print("Looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')

    receiving_matrix = [[] for _ in range(64)]

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    threads = []

    while True:
        sample, timestamp = inlet.pull_sample()
        for row, value in zip(receiving_matrix, sample):
            row.append(value)
        if (len(receiving_matrix[0]) % 721 == 0):
            x = threading.Thread(target=inference, args=(receiving_matrix,info,))
            threads.append(x)
            x.start()
            receiving_matrix = [[] for _ in range(64)]
    
    for _, thread in enumerate(threads):
        thread.join()

if __name__ == '__main__':
    main()
