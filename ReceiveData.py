"""Example program to show how to read a multi-channel time series from LSL."""

from pylsl import StreamInlet, resolve_stream

import time
import numpy as np

import mne
import torch
from torch.utils.data import DataLoader
from models import EEGSegment, MultiStream1DCNN

# Inizio parte relativa al Modello
in_channels = 64
device = "cpu"

# Esempio: usiamo il modello allenato del fold 1
model_inference = MultiStream1DCNN(in_channels=in_channels, n_classes=3).to(device)
model_inference.load_state_dict(torch.load("model_fold_1.pth", map_location=device))
model_inference.eval()
# Fine parte relativa al Modello

def main():
    edf_files = mne.datasets.eegbci.load_data(1, [4])
    raw = mne.io.read_raw_edf(edf_files[0], preload=True, stim_channel='auto', verbose=False)
    info = raw.info
    # first resolve an EEG stream on the lab network
    print("Looking for an EEG stream...")
    streams = resolve_stream('type', 'EEG')
    receiving_matrix = [[] for _ in range(64)]

    # create a new inlet to read from the stream
    inlet = StreamInlet(streams[0])

    while True:
        # get a new sample (you can also omit the timestamp part if you're not
        # interested in it)
        sample, timestamp = inlet.pull_sample()
        print(timestamp)
        for row, value in zip(receiving_matrix, sample):
            row.append(value)
        if (len(receiving_matrix[0]) == 721):
            break
    
    receiving_matrix = np.array(receiving_matrix)
    print(receiving_matrix[0][0])

    segmentoRaw = mne.io.RawArray(receiving_matrix, info)
    segmentoRaw.pick("eeg", exclude='bads') 
    print(type(segmentoRaw))
    print(segmentoRaw.get_data()[0][0])

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
            print("Predizione:", preds.numpy())

if __name__ == '__main__':
    main()
