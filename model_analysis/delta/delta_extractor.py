#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

from dtw_experiment import compute_dtw


def apply_window(x, window_size=None, pool=False, hamming=False):
    n = x.shape[0]
    if window_size is None or n <= window_size:
        window_size = n
    if window_size == n and not pool and not hamming:
        return x
    window = np.concatenate(
        (
            np.zeros((n - window_size) // 2),
            np.hamming(window_size)
            if hamming
            else np.ones(window_size) / window_size,
            np.zeros((n - window_size + 1) // 2),
        )
    )
    if pool:
        return np.tensordot(x, window, axes=(0, 0)).reshape((1, -1))
    return (x * window.reshape(-1, 1))[
        (n - window_size) // 2 : (n - window_size) // 2 + window_size
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "triplets",
        type=Path,
        help="CSV specifying the triplets",
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        type=Path,
        default=Path("."),
        help="directory root of input files",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        help="model name to use for output fields",
    )
    parser.add_argument(
        "-d",
        "--distance",
        choices=["cosine", "kl"],
        default="cosine",
        help="distance function to use",
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
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail if file in triplets is not found",
    )
    args = parser.parse_args()

    header_name = lambda suffix: "_".join(
        filter(None, (args.model_name, suffix))
    )

    sys.stdout.reconfigure(newline="")
    writer = csv.DictWriter(
        sys.stdout,
        (
            "TGT_item",
            "OTH_item",
            "X_item",
        )
        + tuple(
            "_".join(filter(None, (args.model_name, suffix)))
            for suffix in ("TGT", "OTH", "delta")
        ),
        lineterminator=os.linesep,
    )
    writer.writeheader()

    with open(args.triplets, newline="") as f:
        reader = csv.DictReader(f)
        seen = set()
        for row in reader:
            item_to_filename = lambda t: (
                args.input_dir / Path(row[f"{t}_item"])
            ).with_suffix(".npy")
            load_windowed = lambda t: apply_window(
                np.load(item_to_filename(t)),
                window_size=args.width,
                pool=args.pool,
                hamming=args.hamming,
            )

            outrow = {
                "TGT_item": row["TGT_item"],
                "OTH_item": row["OTH_item"],
                "X_item": row["X_item"],
            }
            if tuple(outrow.values()) in seen:
                continue
            seen.add(tuple(outrow.values()))

            try:
                x_rep = load_windowed("X")
                for t in ("TGT", "OTH"):
                    t_rep = load_windowed(t)

                    outrow[header_name(t)] = compute_dtw(
                        t_rep,
                        x_rep,
                        args.distance,
                        norm_div=True,
                    )
                outrow[header_name("delta")] = (
                    outrow[header_name("OTH")] - outrow[header_name("TGT")]
                )

                writer.writerow(outrow)
            except FileNotFoundError as e:
                if args.strict:
                    raise e
                else:
                    print("Warning:", e, file=sys.stderr)
