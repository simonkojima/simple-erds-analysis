import os

from pathlib import Path
import argparse

import mne
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

import pandas as pd

import load

sns.set()


def get_data(tfr, tmin, tmax, band):

    if band == "mu":
        mu = tfr.copy().crop(fmin=8, fmax=12, tmin=tmin, tmax=tmax)
        data = mu
    elif band == "beta":
        beta = tfr.copy().crop(fmin=14, fmax=18, tmin=tmin, tmax=tmax)
        data = beta
    elif band == "gamma":
        gamma = tfr.copy().crop(fmin=36, fmax=40, tmin=tmin, tmax=tmax)
        data = gamma
    else:
        raise ValueError(f"band: {band} is unknown")

    return data


def plot_erds_topo_single(tfr, event, tmin, tmax, fmin, fmax):

    fig, axes = plt.subplots(1, 1, figsize=[10, 10])

    tfr[f"event:{event}"].average().plot_topomap(
        tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, axes=axes, show=False
    )

    return fig


def plot_topo(f_base, save_base, subject, tmin=0.5, tmax=5):
    save_base.mkdir(parents=True, exist_ok=True)

    tfr_acquisition = mne.time_frequency.read_tfrs(
        f_base / f"sub-{subject}_acquisition-tfr.hdf5"
    )
    tfr_online = mne.time_frequency.read_tfrs(f_base / f"sub-{subject}_online-tfr.hdf5")

    if baseline is not None:
        tfr_acquisition.apply_baseline([baseline[0], baseline[1]], mode="percent")
        tfr_online.apply_baseline([baseline[0], baseline[1]], mode="percent")

    for event in ["left", "right"]:
        fig_mu = plot_erds_topo_single(
            tfr_acquisition, event, tmin=tmin, tmax=tmax, fmin=8, fmax=12
        )
        fig_mu.savefig(save_base / f"erds-topo-mu-{event}.png", dpi=300)

        fig_beta = plot_erds_topo_single(
            tfr_acquisition, event, tmin=tmin, tmax=tmax, fmin=14, fmax=18
        )
        fig_beta.savefig(save_base / f"erds-topo-beta-{event}.png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, nargs="*", default="all")
    parser.add_argument("--fmin", type=float, default=2.0)
    parser.add_argument("--fmax", type=float, default=40.0)
    parser.add_argument("--decim", type=int, default=2)
    parser.add_argument("--tmin", type=float, default=-3)
    parser.add_argument("--tmax", type=float, default=5)
    parser.add_argument("--baseline", type=str, nargs="*", default=[-1, 0])
    args = parser.parse_args()

    subject = args.subject
    fmin = args.fmin
    fmax = args.fmax
    decim = args.decim
    tmin = args.tmin
    tmax = args.tmax
    baseline = args.baseline

    if baseline[0] == "None":
        baseline = None
    else:
        baseline = [float(val) for val in baseline]

    data_base = (
        load.config["dir"]["deriv"] / "tfr" / f"fmin-{fmin}_fmax-{fmax}_decim-{decim}"
    )

    if subject == "all":
        subjects = load.config["subjects"]["list"]
    else:
        subjects = subject

    folder_name = "erds-topo"
    if baseline is None:
        folder_name += "-nobaseline"

    for subject in subjects:
        f_base = data_base / subject
        save_base = load.config["dir"]["deriv"] / f"plots" / folder_name / subject

        plot_topo(f_base, save_base, subject)
