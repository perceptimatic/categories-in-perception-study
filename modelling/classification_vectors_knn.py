from math import sqrt
import scipy
import numpy as np
import pandas as pd
import argparse
import os
import glob

from collections import defaultdict


def neighbour_classification_vectors(X_train, X_test, y_train, k, metric="cosine"):
    labels, overall_counts = np.unique(y_train, return_counts=True)
    counts = np.zeros((X_test.shape[0], len(labels)))
    distances = scipy.spatial.distance.cdist(X_test, X_train, metric=metric)
    train_indices = list(range(X_train.shape[0]))
    for i in range(X_test.shape[0]):
        order = sorted(train_indices, key=lambda j: distances[i, j])
        y_i = y_train[order[:k]]
        counts_i = defaultdict(int, dict(zip(*np.unique(y_i, return_counts=True))))
        for i_lab, label in enumerate(labels):
            counts[i, i_lab] = counts_i[label]
    probs = counts / k
    return pd.DataFrame(probs, columns=labels)


def replace_suffix(filename, suffix):
    filename_base = os.path.splitext(filename)[0]
    return filename_base + suffix


def read_features(directory, frames_per_second, rep_filename_suffix, info_table=None):
    X = []
    if info_table is not None:
        files = [
            os.path.join(directory, replace_suffix(f, rep_filename_suffix))
            for f in info_table.loc[:, "filename"]
        ]
    else:
        files = glob.glob(os.path.join(directory, "*" + rep_filename_suffix))

    for i, fn in enumerate(files):
        array = np.load(fn)
        if info_table is not None:
            start_time = info_table.loc[i, "start"]
            end_time = info_table.loc[i, "end"]
        else:
            start_time = 0
            end_time = array.shape[0] / frames_per_second
        duration = end_time - start_time
        if duration > 0.065:
            midpoint = start_time + duration / 2
            duration = 0.065
            start_time = midpoint - duration / 2
        start_frame = round(frames_per_second * start_time)
        duration_frames = round(frames_per_second * duration)
        vowel_array = array[start_frame : (start_frame + duration_frames), :]
        pooled_vowel = np.mean(vowel_array, axis=0)
        X.append(pooled_vowel)
    return files, np.stack(X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("ref_info_file", type=str)
    parser.add_argument("ref_dir", type=str)
    parser.add_argument("test_dir", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("frame_rate", type=int)
    parser.add_argument("k", type=int, nargs="?")
    parser.add_argument(
        "--rep-filename-suffix", type=str, required=False, default="npy"
    )

    args = parser.parse_args()

    ref_info_table = pd.read_csv(args.ref_info_file)
    if args.rep_filename_suffix[0] != ".":
        args.rep_filename_suffix = "." + args.rep_filename_suffix

    fns_ref, X_ref = read_features(
        args.ref_dir, args.frame_rate, args.rep_filename_suffix, ref_info_table
    )
    fns_test, X_test = read_features(
        args.test_dir, args.frame_rate, args.rep_filename_suffix
    )

    k = args.k
    if k is None:
        k = int(sqrt(X_ref.shape[0]))
    classification_vectors = neighbour_classification_vectors(
        X_ref, X_test, ref_info_table.phone_label, k
    )
    classification_vectors["filename"] = [
        replace_suffix(os.path.basename(f), "") for f in fns_test
    ]
    classification_vectors.to_csv(args.output_file, index=False)
