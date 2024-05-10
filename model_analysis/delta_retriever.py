#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 21:49:24 2023

@author: PaulaArkhangorodsky, RobinHuo
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from dtw_experiment import compute_dtw


def column_name(model_name, data_type_str):
    return model_name + "_" + data_type_str


def apply_window(x, window_size=None, pool=False, rect=False):
    n = x.shape[0]
    if window_size is None or n <= window_size:
        window_size = n
    if window_size == n and not pool and rect:
        return x
    window = np.concatenate(
        (
            np.zeros((n - window_size) // 2),
            np.ones(window_size) / window_size
            if rect
            else np.hamming(window_size),
            np.zeros((n - window_size + 1) // 2),
        )
    )
    if pool:
        return np.tensordot(x, window, axes=(0, 0)).reshape((1, -1))
    return (x * window.reshape(-1, 1))[
        (n - window_size) // 2 : (n - window_size) // 2 + window_size
    ]


if __name__ == "__main__":
    # Read in command-line arguments
    parser = argparse.ArgumentParser(
        description="Calculate deltas for the model"
    )

    parser.add_argument(
        "-d",
        "--directories",
        type=str,
        nargs="+",
        required=True,
        help="list of directories where resp. model files are stored",
    )
    parser.add_argument(
        "-s",
        "--distances",
        type=str,
        nargs="+",
        required=True,
        help="list of distances that each resp. model's deltas will use",
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        required=True,
        help="name of resulting csv file",
    )
    parser.add_argument(
        "-e",
        "--experimental",
        type=str,
        required=True,
        help="name of experimental data csv file",
    )
    parser.add_argument(
        "-m",
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="list of model names",
    )
    parser.add_argument(
        "-x",
        "--extension",
        default="wav",
        choices=["wav", "aif"],
        type=str,
        help="extension of the audio files",
    )
    parser.add_argument(
        "--hamming",
        action="store_true",
        help="apply Hamming window to model representations",
    )
    parser.add_argument(
        "--width", type=int, default=None, help="width of window to apply"
    )
    parser.add_argument(
        "--pool", action="store_true", help="sum-reduce frames in window"
    )

    commandline_input = parser.parse_args()

    model_list = commandline_input.models
    directory_list = commandline_input.directories
    distance_list = commandline_input.distances
    extension = "." + commandline_input.extension

    infile = open(commandline_input.experimental, newline="")
    reader = csv.reader(infile)

    outfile = open(commandline_input.outfile, "w", newline="")
    writer = csv.writer(outfile)

    # Store header and write output header.
    header = next(reader)
    out_header = header[:]
    for model in model_list:
        out_header.extend(
            column_name(model, t) for t in ("tgt", "oth", "delta")
        )
    writer.writerow(out_header)

    count = 0
    # Create distance cache for each model.
    model_caches = [{} for _ in range(len(model_list))]
    for row in reader:
        to_write = row[:]
        for model_i, model in enumerate(model_list):
            directory = Path(directory_list[model_i])
            distance = distance_list[model_i]
            cache = model_caches[model_i]

            dists = {
                "TGT": None,
                "OTH": None,
            }

            row_fields = {col_name: val for col_name, val in zip(header, row)}

            # Representation filenames to load.
            filenames = {
                t: row_fields[f"{t}_item"].removesuffix(extension) + ".npy"
                for t in ("TGT", "OTH", "X")
            }

            # Get distances from cache or compute if not cached.
            for t in ("TGT", "OTH"):
                if (filenames[t], filenames["X"]) not in cache:
                    t_rep = np.load(directory / filenames[t])
                    x_rep = np.load(directory / filenames["X"])

                    window_size = commandline_input.width
                    pool = commandline_input.pool
                    rect = not commandline_input.hamming
                    t_rep = apply_window(t_rep, window_size, pool, rect)
                    x_rep = apply_window(x_rep, window_size, pool, rect)

                    cache[(filenames[t], filenames["X"])] = compute_dtw(
                        t_rep,
                        x_rep,
                        distance,
                        norm_div=True,
                    )
                dists[t] = cache[(filenames[t], filenames["X"])]

            to_write.extend(
                [dists["TGT"], dists["OTH"], dists["OTH"] - dists["TGT"]]
            )

        writer.writerow(to_write)
        count += 1
        if count % 100 == 0:
            print(count)

    infile.close()
    outfile.close()

    print("Done!")
