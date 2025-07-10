import os

from pathlib import Path
import mne

import tag_mne as tm

l_freq = 8
h_freq = 30
tmin = -5
tmax = 7

subjects = ["A1", "A2", "A3", "A4", "A5"]

raw_base = Path("~/Documents/datasets/dreyer_2023/raw")

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

# load raw data
for subject in subjects:
    