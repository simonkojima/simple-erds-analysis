import os

from pathlib import Path
import argparse

import mne
import numpy as np

import tag_mne as tm

import load

def get_files(f_base, subject, keys = dict(acquisition = 'acquisition', online = 'onlineT')):
    
    files = os.listdir(f_base)
    files.sort()
    
    acquisition = list()
    online = list()
    
    for file in files:
        if keys['acquisition'] in file:
            acquisition.append(file)
        elif keys['online'] in file:
            online.append(file)
    
    return acquisition, online

def epochs_from_files(f_base,
                      files,
                      l_freq,
                      h_freq,
                      subject):
    epochs_list = list()
    for file in files:
        run = int(file.split("_")[1][1])
        rtype = file.split("_")[2].split(".")[0]
        
        raw = mne.io.read_raw_gdf(f_base / file,
                                  preload = True)

        raw.filter(l_freq = l_freq,
                   h_freq = h_freq,
                   method = 'iir',
                   iir_params = {'ftype':'butter',
                                 'order':2,
                                 'btype':'bandpass'})

        # eog and emg mapping
        mapping = dict()
        for ch in raw.ch_names:
            if "EOG" in ch:
                mapping[ch] = 'eog'
            elif "EMG" in ch:
                mapping[ch] = 'emg'
        
        raw.set_channel_types(mapping)
        raw.set_montage("standard_1020")

        events, event_id = mne.events_from_annotations(raw)
        
        samples, markers =  tm.markers_from_events(events, event_id)
        markers = tm.add_tag(markers, f"subject:{subject}")
        markers = tm.add_event_names(markers, {'left': ['769'], 'right':['770']})
        markers = tm.add_tag(markers, f"run:{run}")
        markers = tm.add_tag(markers, f"rtype:{rtype}")

        samples, markers = tm.remove(samples, markers, "event:misc")
        
        events, event_id = tm.events_from_markers(samples, markers)
        epochs = mne.Epochs(raw = raw,
                            tmin = tmin,
                            tmax = tmax,
                            events = events,
                            event_id = event_id,
                            baseline = None)

        epochs_list.append(epochs) 
    
    epochs = tm.concatenate_epochs(epochs_list)
    
    return epochs

def main(f_base, save_base, subject, l_freq, h_freq, resample):
    
    files_acquisition, files_online = get_files(f_base, subject)
    
    l_freq = float(l_freq)
    h_freq = float(h_freq)

    epochs_acquisition = epochs_from_files(f_base, files_acquisition, l_freq = l_freq, h_freq = h_freq, subject = subject)
    epochs_online = epochs_from_files(f_base, files_online, l_freq = l_freq, h_freq = h_freq, subject = subject)
    
    epochs_acquisition.resample(resample)
    epochs_online.resample(resample)
    
    save_base.mkdir(parents = True, exist_ok = True)
    
    epochs_acquisition.save(save_base / f"sub-{subject}_acquisition-epo.fif", overwrite = True)
    epochs_online.save(save_base / f"sub-{subject}_online-epo.fif", overwrite = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--l_freq", type = float, default = 1.0)
    parser.add_argument("--h_freq", type = float, default = 45.0)
    parser.add_argument("--tmin", type = float, default = -5)
    parser.add_argument("--tmax", type = float, default = 7)
    parser.add_argument("--resample", type = float, default = 128)
    args = parser.parse_args()

    l_freq = args.l_freq
    h_freq = args.h_freq
    tmin = args.tmin
    tmax = args.tmax
    resample = args.resample

    data_base = load.config['dir']['raw']
    subjects = load.config['subjects']['list']
    
    print(subjects)
    
    for subject in subjects:
        f_base = data_base / subject
        save_base = load.config['dir']['deriv'] / "epochs" / f"l_freq-{l_freq}_h_freq-{h_freq}_resample-{resample}" / subject

        main(f_base, save_base, subject, l_freq, h_freq, resample)