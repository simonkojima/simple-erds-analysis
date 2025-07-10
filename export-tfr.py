import os
import argparse
import numpy as np

from pathlib import Path

import mne

import load

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--l_freq", type = float, default = 1.0)
    parser.add_argument("--h_freq", type = float, default = 45.0)
    parser.add_argument("--fmin", type = float, default = 2.0)
    parser.add_argument("--fmax", type = float, default = 40.0)
    parser.add_argument("--decim", type = int, default = 2)
    parser.add_argument("--resample", type = float, default = 128)
    args = parser.parse_args()

    l_freq = args.l_freq
    h_freq = args.h_freq
    fmin = args.fmin
    fmax = args.fmax
    decim = args.decim
    resample = args.resample
    
    data_base = load.config['dir']['deriv'] / "epochs"/ f"l_freq-{l_freq}_h_freq-{h_freq}_resample-{resample}"
    subjects = load.config['subjects']['list']
    save_base = load.config['dir']['deriv'] / "tfr" / f"fmin-{fmin}_fmax-{fmax}_decim-{decim}"
    
    for subject in subjects:

        epochs_acquisition = mne.read_epochs(data_base / subject / f"sub-{subject}_acquisition-epo.fif")
        epochs_online = mne.read_epochs(data_base / subject / f"sub-{subject}_online-epo.fif")
        
        tfr_acquisition = epochs_acquisition.compute_tfr(
            method="multitaper",
            freqs=np.arange(fmin, fmax + 1),
            n_cycles=np.arange(fmin, fmax + 1),
            use_fft=True,
            return_itc=False,
            average=False,
            decim=decim,
            n_jobs=-1,
        )

        tfr_online = epochs_online.compute_tfr(
            method="multitaper",
            freqs=np.arange(fmin, fmax + 1),
            n_cycles=np.arange(fmin, fmax + 1),
            use_fft=True,
            return_itc=False,
            average=False,
            decim=decim,
            n_jobs=-1,
        )
        
        (save_base / subject).mkdir(parents = True, exist_ok=True)
        
        tfr_acquisition.save(save_base / subject / f"sub-{subject}_acquisition-tfr.hdf5", overwrite = True)
        tfr_online.save(save_base / subject / f"sub-{subject}_online-tfr.hdf5", overwrite = True)