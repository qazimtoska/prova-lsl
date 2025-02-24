import os
import numpy as np
import mne
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import KFold
from tqdm import tqdm

# Per TensorBoard
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

# Per gestione asincrona delle scritture dei comandi su seriale
import threading
import queue
import time

# Per l'invio dei comandi tramite seriale
from periphery import Serial

uart1 = Serial("/dev/ttymxc0", 115200)
serial_queue = queue.Queue()

# Comandi da inviare all'ESP32
FORWARD = "AT+FORWARD={}\n"
BACK = "AT+BACK={}\n"
LEFT = "AT+LEFT={}\n"
RIGHT = "AT+RIGHT={}\n"
STOP = "AT+STOP={}\n"

SUBJECTS = [1]
RUNS = [4, 8, 12]

# Hyperparametri e configurazioni
k_folds = 10
batch_size = 8
learning_rate = 1e-4
weight_decay = 1e-4
num_epochs = 20  # puoi aumentare se hai più tempo/hardware

device = "cpu"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Utilizzeremo il device:", device)

def send_commands_through_serial(labels):
    for label in labels:
        if label == 0:
            command = STOP.format(127)
        elif label == 1:
            command = LEFT.format(127)
        elif label == 2:
            command = RIGHT.format(127)
        elif label == 3:
            command = FORWARD.format(127)
        else:
            command = BACK.format(127)
        uart1.write(command.encode('utf-8'))

def serial_worker():
    while True:
        labels = serial_queue.get()  # Blocca finché non c'è qualcosa in coda
        if labels is None:
            break
        send_commands_through_serial(labels)
        serial_queue.task_done()

class PhysioNet3ClassDataset(Dataset):
    """
    Dataset EEG Motor Imagery per un singolo soggetto e un insieme di 'run'.
    """
    def __init__(self, subject_id, runs, tmin=-0.5, tmax=4.0):
        """
        Carica i file EDF di un singolo soggetto, crea epoche e converte in tensori.

        :param subject_id: int, id soggetto (1..109, skip alcuni noti)
        :param runs: list, i numeri dei run che contengono T0=1, T1=2, T2=3
        :param tmin: float, inizio epoca in secondi
        :param tmax: float, fine epoca in secondi
        """
        # Carica i file EDF per questo soggetto e i run scelti
        edf_files = mne.datasets.eegbci.load_data(subject_id, runs)

        raws = []
        for ef in edf_files:
            raw = mne.io.read_raw_edf(ef, preload=True, stim_channel='auto', verbose=False)
            print(raw.info)
            raws.append(raw)

        # Concatena i raw in un unico oggetto
        self.raw = mne.concatenate_raws(raws)

        # print(self.raw.get_data()[0][592])
        # Pick channels
        self.raw.pick("eeg", exclude='bads')

        # Trova gli eventi (annotazioni)
        events, event_id_dict = mne.events_from_annotations(self.raw, verbose=False)
        print(events)
        # Manteniamo solo i 3 tipi di evento che ci interessano
        desired_ids = dict(T0=1, T1=2, T2=3)

        # Creiamo le epoche
        epochs = mne.Epochs(
            self.raw,
            events,
            event_id=desired_ids,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False
        )

        # Convertiamo da Volts a microVolts
        data = epochs.get_data() * 1e6  # shape (n_epochs, n_channels, n_times)
        # print(data[0][0][0])
        # print(data)
        # Etichette (1,2,3) -> (0,1,2)
        labels = epochs.events[:, -1]   # {1,2,3}
        labels = labels - 1            # => {0,1,2}

        # Creiamo i tensori PyTorch
        self.X = torch.tensor(data, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

        # print(f"[Soggetto {subject_id}] X={self.X.shape}, y={self.y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

"""
subject_datasets = []
for subj in SUBJECTS:
    ds = PhysioNet3ClassDataset(subj, RUNS, tmin=-0.5, tmax=4.0)
    subject_datasets.append(ds)
"""

# Numero di canali, es. 64
# in_channels = subject_datasets[0].X.shape[1]
in_channels = 64

class MultiStream1DCNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_classes: int = 3,
        n_streams: int = 4,
        start_kernel: int = 7,
        stream_depth: int = 2,
        pool_output_size: int = 32,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.n_classes   = n_classes
        self.n_streams   = n_streams
        self.start_kernel= start_kernel
        self.stream_depth= stream_depth
        self.pool_output_size = pool_output_size

        # Costruiamo gli stream
        streams = []
        for i in range(n_streams):
            k_size = start_kernel + 2*i  # es. 7,9,11,13
            stream = self._build_stream(in_channels, k_size, stream_depth)
            streams.append(stream)
        self.streams = nn.ModuleList(streams)

        # Pooling adattivo per uniformare la lunghezza temporale a pool_output_size
        self.adapool = nn.AdaptiveMaxPool1d(pool_output_size)

        # Calcoliamo il numero di feature in input al classificatore
        # Ogni stream ha 64 canali in output, dimensione tempo = pool_output_size
        in_feats = 64 * pool_output_size * n_streams
        hidden_dim = 256

        # Classificatore fully-connected
        self.classifier = nn.Sequential(
            nn.Linear(in_feats, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, n_classes),
        )

    def _build_stream(self, in_ch, kernel_size, depth):
        """
        Crea una serie di blocchi Conv1d->ReLU->Conv1d->ReLU->MaxPool, ripetuti 'depth' volte.
        """
        layers = []
        c_in = in_ch
        for _ in range(depth):
            c_out = 64
            layers.append(nn.Conv1d(c_in, c_out, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.Conv1d(c_out, c_out, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
            c_in = c_out
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        x: [B, in_channels, time]
        """
        outs = []
        for stream in self.streams:
            s_out = stream(x)            # => [B,64, time_ridotta]
            s_out = self.adapool(s_out)  # => [B,64,32]
            s_out = s_out.view(s_out.size(0), -1)  # => [B, 64*32]
            outs.append(s_out)
        # Concateniamo le feature di tutti gli stream
        concat = torch.cat(outs, dim=1)   # => [B, 64*32*n_streams]
        # Classificazione finale
        logits = self.classifier(concat)  # => [B, 3]
        return logits

# Esempio: usiamo il modello allenato del fold 1
model_inference = MultiStream1DCNN(in_channels=in_channels, n_classes=3).to(device)
model_inference.load_state_dict(torch.load("model_fold_1.pth", map_location=device))
model_inference.eval()

# Carichiamo un piccolo dataset di esempio (ad es. soggetto 1, run [4])
example_dataset = PhysioNet3ClassDataset(subject_id=1, runs=[4], tmin=-0.5, tmax=4.0)
example_loader = DataLoader(example_dataset, batch_size=batch_size, shuffle=False)

# Creazione del background worker
worker_thread = threading.Thread(target=serial_worker, daemon=True)
worker_thread.start()

correct, total = 0, 0
with torch.no_grad():
    all_preds = []      # Qui salviamo le classi predette
    all_targets = []    # Qui salviamo le classi effettive
    for xb, yb in example_loader:
        xb, yb = xb.to(device), yb.to(device)
        
        logits = model_inference(xb)
        _, preds = logits.max(dim=1)

        serial_queue.put(preds.numpy()) # Inserimento delle predizioni nella coda

        all_preds.extend(preds)
        all_targets.extend(yb)

        correct += preds.eq(yb).sum().item()
        total   += xb.size(0)

# Attendo che tutti gli elementi nella coda siano processati
serial_queue.join()

serial_queue.put(None)
worker_thread.join()

inference_acc = 100.0 * correct / total
print(f"\nAccuracy su soggetto=1, run=[4], con i pesi caricati: {inference_acc:.2f}%")
print(f"Predicted classes: {[int(x) for x in all_preds]}")
print(f"Actual classes:    {[int(x) for x in all_targets]}")

uart1.close()
