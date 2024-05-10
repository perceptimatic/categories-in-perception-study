#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created march 2021
    by Juliette MILLET
    script to compute mfccs or other acoustic features and save them
    with the same structure than the original dataset
"""

import os
from pathlib import Path

import librosa
import numpy as np


def compute_mfccs(filename):
    y, sr = librosa.load(filename, sr=None)
    spect = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13,
        win_length=int(0.025 * sr),
        hop_length=int(0.010 * sr),
    )

    spect = spect.T
    return spect


def compute_melfilterbanks(filename):
    y, sr = librosa.load(filename, sr=None)
    spect = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        win_length=int(0.025 * sr),
        hop_length=int(0.010 * sr),
    )
    spect = librosa.amplitude_to_db(spect)
    spect = spect.T
    return spect


def transform_and_save(filename_in, filename_out, features):
    if features == "mfccs":
        spect = compute_mfccs(filename_in)
        np.save(filename_out, spect)
    elif features == "melfilterbanks":
        spect = compute_melfilterbanks(filename_in)
        np.save(filename_out, spect)
    else:
        print("The feature you asked for is not available")
        raise ValueError


def transform_all(folder_in, folder_out, features):
    for dirname, _, filenames in os.walk(folder_in):
        for filename in filenames:
            if Path(filename).suffix not in [".wav", ".flac"]:
                continue
            relpath = (Path(dirname) / filename).relative_to(folder_in)
            os.makedirs(folder_out / relpath.parent, exist_ok=True)

            transform_and_save(
                folder_in / relpath,
                folder_out / relpath.with_suffix(".npy"),
                features,
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="script to analyze output from discrimination experiment"
    )
    parser.add_argument(
        "folder_perceptimatic",
        metavar="in",
        type=Path,
        help="folder where input are",
    )
    parser.add_argument(
        "folder_out",
        metavar="out",
        type=Path,
        help="folder where to put outputs",
    )
    parser.add_argument("-f", "--features", choices=["mfccs", "melfilterbanks"], default="mfccs")

    args = parser.parse_args()

    transform_all(
        folder_in=args.folder_perceptimatic,
        folder_out=args.folder_out,
        features=args.features,
    )
