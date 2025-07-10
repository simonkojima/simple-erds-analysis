import os
import json
import argparse
import numpy as np

from pathlib import Path

import pandas as pd

import mne

import load

def compute_mask(tfr, alpha = 0.05):
    kwargs = dict(n_permutations=100, step_down_p=0.05, seed=42, buffer_size=None, out_type="mask")  # for cluster test
    
    ch_names = tfr.ch_names
    
    masks = list()
    for idx_ch, ch in enumerate(ch_names):
        
        _, c1, p1, _ = mne.stats.permutation_cluster_1samp_test(tfr.data[:, idx_ch, :, :], tail = 1, **kwargs, n_jobs = -1)
        _, c2, p2, _ = mne.stats.permutation_cluster_1samp_test(tfr.data[:, idx_ch, :, :], tail = -1, **kwargs, n_jobs = -1)

        c = np.stack(c1 + c2, axis=2)  # combined clusters
        p = np.concatenate((p1, p2))  # combined p-values
        mask = c[..., p <= 0.05].any(axis=-1)
        masks.append(mask)
        
    masks = np.array(masks)
    return masks

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fmin", type = float, default = 2.0)
    parser.add_argument("--fmax", type = float, default = 40.0)
    parser.add_argument("--decim", type = int, default = 2)
    parser.add_argument("--baseline", type = str, nargs = '*', default = [-1, 0])
    args = parser.parse_args()

    fmin = args.fmin
    fmax = args.fmax
    decim = args.decim
    baseline = args.baseline
    
    if baseline[0] == 'None':
        baseline = None
    else:
        baseline = [float(val) for val in baseline]
    
    data_base = load.config['dir']['deriv'] / "tfr" / f"fmin-{fmin}_fmax-{fmax}_decim-{decim}"
    subjects = load.config['subjects']['list']
    save_base = load.config['dir']['deriv'] / "mask" / f"fmin-{fmin}_fmax-{fmax}_decim-{decim}"

    for subject in subjects:
        f_base = data_base / subject

        tfr_acquisition = mne.time_frequency.read_tfrs(f_base / f"sub-{subject}_acquisition-tfr.hdf5")
        tfr_online = mne.time_frequency.read_tfrs(f_base / f"sub-{subject}_online-tfr.hdf5")
        
        if baseline is not None:
            tfr_acquisition.apply_baseline([baseline[0], baseline[1]], mode = 'percent')
            tfr_online.apply_baseline([baseline[0], baseline[1]], mode = 'percent')
        
        df_acquisition = pd.DataFrame()
        df_acquisition['event'] = ['left', 'right']
        df_acquisition['ch_names'] = [tfr_acquisition.ch_names for m in range(2)]
        df_acquisition['mask'] = [compute_mask(tfr_acquisition['event:left']), compute_mask(tfr_acquisition['event:right'])]

        df_online = pd.DataFrame()
        df_online['event'] = ['left', 'right']
        df_online['ch_names'] = [tfr_online.ch_names for m in range(2)]
        df_online['mask'] = [compute_mask(tfr_online['event:left']), compute_mask(tfr_online['event:right'])] 

        (save_base / subject).mkdir(parents = True, exist_ok = True)
        
        df_acquisition.to_pickle(save_base / subject / "mask-acquisition.pkl")
        df_acquisition.to_html(save_base / subject / "mask-acquisition.html")

        df_online.to_pickle(save_base / subject / "mask-online.pkl")
        df_online.to_html(save_base / subject / "mask-online.html")
                                  
