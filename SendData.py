"""Example program to demonstrate how to send a multi-channel time series to
LSL."""

"""
Per capire come inviare il segmento:
1. La prima riga utile negli eventi non è la 0, ma la 1 poichè la riga 0 parte dal campione 0 e non si può andare indietro di 0.5 secondi
2. Quando individuo una riga, il range lo ottengo facendo[(num_campione - 80) ; (num_campione + 640 + 1)]
   Note: 80 lo ottengo come 160 * 0.5 secondi
         640 lo ottengo come 160 * 4
         L'1 consente di prendere in totale 721 campioni e non 720 (come vuole mne)
"""
import sys
import getopt

import time
from random import random as rand

from pylsl import StreamInfo, StreamOutlet, local_clock

import mne

def main(argv):
    srate = 160
    name = 'BioSemi'
    type = 'EEG'
    n_channels = 64
    help_string = 'SendData.py -s <sampling_rate> -n <stream_name> -t <stream_type>'

    edf_files = mne.datasets.eegbci.load_data(1, [4]) # soggetto 1, run 4
    raw = mne.io.read_raw_edf(edf_files[0], preload=True, stim_channel='auto', verbose=False)
    raw_matrix = raw.get_data()
    # segment = [inner_list[592:1314] for inner_list in raw_matrix]
    print(time.time())
    segment = [inner_list[1248:1969] for inner_list in raw_matrix]
    print(time.time())

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

    print("now sending data...")
    start_time = local_clock()
    sent_samples = 0

    while True:
        time.sleep(10)
        for col_index in range(721):
            column = [row[col_index] for row in segment]
            outlet.push_sample(column)
            time.sleep(0.00625)
        break

if __name__ == '__main__':
    main(sys.argv[1:])
