import os

from pathlib import Path
import argparse

import mne
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

import pandas as pd

import constants
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


def plot_erds_map_single(tfr, event, mask):

    tfr = tfr[f"event:{event}"].copy()

    ch_names = tfr.ch_names

    n_horizontal = 7
    n_vertical = 5
    vmin, vmax = -1, 1.5  # set min and max ERDS values in plot
    cnorm = matplotlib.colors.TwoSlopeNorm(
        vmin=vmin, vcenter=0, vmax=vmax
    )  # min, center & max ERDS

    width_ratios = [10 for m in range(n_horizontal)]
    fig, axes = plt.subplots(
        n_vertical,
        n_horizontal,
        figsize=(20, 10),
        gridspec_kw={"width_ratios": width_ratios},
    )

    for idx_ch, ch_name in enumerate(ch_names):

        map = constants.map_27ch[ch_name]
        ax = axes[map[0], map[1]]

        if mask is not None:
            mask_nd = mask[mask["event"] == event]["mask"].to_numpy()[0]
            idx = mask[mask["event"] == event]["ch_names"].to_list()[0].index(ch_name)
            mask_nd = mask_nd[idx, :, :]

            tfr.average().plot(
                [idx_ch],
                cmap="RdBu",
                cnorm=cnorm,
                axes=ax,
                colorbar=False,
                show=False,
                mask=mask_nd,
                mask_style="mask",
            )
        else:
            tfr.average().plot(
                [idx_ch],
                cmap="RdBu",
                cnorm=cnorm,
                axes=ax,
                colorbar=False,
                show=False,
            )

        ax.set_title(ch_name)
        ax.set_xlim([tmin, tmax])

    plt.suptitle(f"{event}")
    plt.tight_layout()

    return fig


def plot_erds_map(f_base, save_base, subject):
    save_base.mkdir(parents=True, exist_ok=True)

    tfr_acquisition = mne.time_frequency.read_tfrs(
        f_base / f"sub-{subject}_acquisition-tfr.hdf5"
    )
    tfr_online = mne.time_frequency.read_tfrs(f_base / f"sub-{subject}_online-tfr.hdf5")

    if baseline is not None:
        tfr_acquisition.apply_baseline([baseline[0], baseline[1]], mode="percent")
        tfr_online.apply_baseline([baseline[0], baseline[1]], mode="percent")

    for event in ["left", "right"]:
        if nomask is False:
            mask = pd.read_pickle(
                load.config["dir"]["deriv"]
                / "mask"
                / f"fmin-{fmin}_fmax-{fmax}_decim-{decim}"
                / subject
                / "mask-acquisition.pkl"
            )
        else:
            mask = None
        fig = plot_erds_map_single(tfr_acquisition, event, mask)
        fig.savefig(save_base / f"acquisition_{event}.png", dpi=300)
        plt.close()

        if nomask is False:
            mask = pd.read_pickle(
                load.config["dir"]["deriv"]
                / "mask"
                / f"fmin-{fmin}_fmax-{fmax}_decim-{decim}"
                / subject
                / "mask-online.pkl"
            )
        else:
            mask = None
        fig = plot_erds_map_single(tfr_online, event, mask)
        fig.savefig(save_base / f"online_{event}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=str, nargs="*", default="all")
    parser.add_argument("--fmin", type=float, default=2.0)
    parser.add_argument("--fmax", type=float, default=40.0)
    parser.add_argument("--decim", type=int, default=2)
    parser.add_argument("--tmin", type=float, default=-3)
    parser.add_argument("--tmax", type=float, default=5)
    parser.add_argument("--baseline", type=str, nargs="*", default=[-1, 0])
    parser.add_argument("--nomask", action="store_true")
    args = parser.parse_args()

    subject = args.subject
    fmin = args.fmin
    fmax = args.fmax
    decim = args.decim
    tmin = args.tmin
    tmax = args.tmax
    baseline = args.baseline
    nomask = args.nomask

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

    folder_name = "erds-map"
    if baseline is None:
        folder_name += "-nobaseline"
    if nomask:
        folder_name += "-nomask"

    for subject in subjects:
        f_base = data_base / subject
        save_base = load.config["dir"]["deriv"] / f"plots" / folder_name / subject

        # print(save_base)

        # main(f_base, save_base, subject, l_freq, h_freq, resample)

        plot_erds_map(f_base, save_base, subject)
