"""Example program to demonstrate how to send a multi-channel time series to
LSL."""

import sys
import getopt

import time
from random import random as rand

from pylsl import StreamInfo, StreamOutlet, local_clock

import mne

def main(argv):
    # Costanti e inizializzazione LSL
    
    srate = 160
    name = 'BioSemi'
    type = 'EEG'
    n_channels = 64
    help_string = 'SendData.py -s <sampling_rate> -n <stream_name> -t <stream_type>'

    try:
        opts, args = getopt.getopt(argv, "hs:c:n:t:", longopts=["srate=", "channels=", "name=", "type"])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(help_string)
            sys.exit()
        elif opt in ("-s", "--srate"):
            srate = float(arg)
        elif opt in ("-c", "--channels"):
            n_channels = int(arg)
        elif opt in ("-n", "--name"):
            name = arg
        elif opt in ("-t", "--type"):
            type = arg

    # first create a new stream info (here we set the name to BioSemi,
    # the content-type to EEG, 8 channels, 100 Hz, and float-valued data) The
    # last value would be the serial number of the device or some other more or
    # less locally unique identifier for the stream as far as available (you
    # could also omit it but interrupted connections wouldn't auto-recover)
    info = StreamInfo(name, type, n_channels, srate, 2, 'myuid34234')

    # next make an outlet
    outlet = StreamOutlet(info)

    # Apertura file di registrazione

    # edf_files contiene i path delle registrazioni, in questo caso solo una
    edf_files = mne.datasets.eegbci.load_data(1, [4]) # soggetto 1, run 4
    
    # raw è un'istanza di RawEDF, e viene creato a partire dalla registrazione scelta al passo precedente
    raw = mne.io.read_raw_edf(edf_files[0], preload=True, stim_channel='auto', verbose=False)

    # raw_matrix contiene i dati della registrazione sotto forma di matrice (n_channels X n_total_samples)
    raw_matrix = raw.get_data()

    # signal_length rappresenta la lunghezza del segnale, in termini di campioni 
    signal_length = len(raw_matrix[0])

    print("now sending data...")

    while not outlet.have_consumers():
        print("Waiting for consumers...")
        time.sleep(1)

    while True:
        for col_index in range(signal_length):
            # invio una colonna alla volta della matrice raw_matrix
            column = [row[col_index] for row in raw_matrix]
            outlet.push_sample(column)
            time.sleep(0.00625) # simulo un campionamento a 160 Hz -> 1/160 = 0.00625 secondi

if __name__ == '__main__':
    main(sys.argv[1:])
