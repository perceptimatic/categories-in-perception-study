#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created march 2021
    by Juliette MILLET
    script to compute mfccs or other acoustic features and save them
    with the same structure than the original dataset
"""
import librosa
import os
import numpy as np

def compute_mfccs(filename):
    y, sr = librosa.load(filename)
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
    y, sr = librosa.load(filename)
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
    if features == 'mfccs':
        spect = compute_mfccs(filename_in)
        np.save(filename_out, spect)
    else if features == 'melfilterbanks':
        spect = compute_melfilterbanks(filename_in)
        np.save(filename_out, spect)
    else:
        print('The feature you asked for is not available')
        raise ValueError


def transform_all(folder_in, folder_out, features):
    for name in os.listdir(folder_in):
        a = os.path.join(folder_in, name)
        if not a.endswith('.wav'):
            continue
        path_output = os.path.join(folder_out, name).replace('.wav', '.npy')
        transform_and_save(a, path_output, features)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='script to compute mfccs')
    parser.add_argument('folder_wavs', type=str,
                        help='folder where the wav files to transform are')
    parser.add_argument('folder_out', type=str,
                        help='folder where to put representations')
    parser.add_argument('features', type=str,
                        help='features wanted mfccs or melfilterbanks')
    args = parser.parse_args()

    transform_all(folder_in=args.folder_wavs, folder_out=args.folder_out, features=args.features)

