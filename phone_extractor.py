#!/usr/bin/env python3
"""
Extract segments from representations given TIMIT-style alignments or a CSV
manifest.

If given a CSV manifest, it must have the fields start, end, and path.
If path is relative, an input root directory must be provided with -i.

Example:
python phone_extractor.py manifest.csv -i $REPRESENTATIONS/ -d 10 -f '{stem}.npy'

Author: RobinHuo
"""
import argparse
import csv
import os
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd


def make_segment(start: int, end: int, phone: Optional[str] = None) -> dict:
    if phone is None:
        return {
            "start": start,
            "end": end,
        }
    else:
        return {
            "start": start,
            "end": end,
            "phone": phone,
        }


def iter_alignments_from_dir(root: Path):
    for trans_file in root.glob("**/*.phn"):
        with open(trans_file) as f:
            yield trans_file.with_suffix(".npy").relative_to(root), [
                make_segment(
                    int(tokens[0]),
                    int(tokens[1]),
                    tokens[2] if len(tokens) > 2 else None,
                )
                for tokens in map(lambda line: line.split(), f)
            ]


def iter_alignments_from_csv(csv_path: Path):
    table = pd.read_csv(csv_path).groupby(by="path", as_index=True)
    for path, group in table:
        yield Path(path), [
            make_segment(
                row["start"],
                row["end"],
                row["phone"] if "phone" in row else None,
            )
            for _, row in group.iterrows()
        ]


def extract_phones(
    alignments: Iterable,
    root: Optional[Path] = None,
    out_root: Optional[Path] = None,
    out_csv_path: Optional[Path] = None,
    frame_transform: Optional[Callable] = None,
    format_str: Optional[str] = None,
):
    if out_root is None and out_csv_path is None:
        return

    if frame_transform is None:
        frame_transform = lambda x: x

    out_csv = None
    if out_csv_path is not None:
        out_csv = open(out_csv_path, "w", newline="")
        writer = csv.DictWriter(
            out_csv, fieldnames=["phone", "start", "end", "path"]
        )

    try:
        if out_csv is not None:
            writer.writeheader()

        if out_root is not None and not os.path.isdir(out_root):
            os.mkdir(out_root)

        for path, segments in alignments:
            rep_path = root / path if not path.is_absolute() else path
            if not path.is_absolute():
                relpath = path
            elif root is not None:
                relpath = path.relative_to(root)
            else:
                relpath = Path(path.name)

            segment_transform = lambda segment: {
                "start": frame_transform(segment["start"]),
                "end": frame_transform(segment["end"]),
                "phone": segment.get("phone"),
                "path": rep_path,
            }

            if out_csv is not None:
                for segment in map(segment_transform, segments):
                    writer.writerow(segment)

            if out_root is not None:
                for segment in map(segment_transform, segments):
                    if segment["start"] >= segment["end"]:
                        continue
                    rep = np.load(rep_path)[segment["start"] : segment["end"]]
                    out_name = format_str.format(
                        start=segment["start"],
                        end=segment["end"],
                        phone=segment["phone"],
                        stem=relpath.stem,
                        parent=str(relpath.parent),
                    )
                    out_path = out_root / out_name
                    os.makedirs(out_path.parent, exist_ok=True)
                    np.save(out_path, rep)

    finally:
        if out_csv is not None:
            out_csv.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "alignments",
        type=Path,
        help="CSV manifest of segments or root dir of TIMIT-style alignments",
    )
    parser.add_argument(
        "-i",
        "--root",
        "--representations-root",
        type=Path,
        help="root directory of representations",
    )
    parser.add_argument(
        "-d",
        "--downsample",
        type=float,
        default=1.0,
        help="downsample alignments by this factor before applying",
    )
    parser.add_argument(
        "-u",
        "--upsample",
        type=float,
        default=1.0,
        help="upsample alignments by this factor before applying",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        "--output-root",
        type=Path,
        help="root directory to place extracted segments in",
    )
    parser.add_argument(
        "-c",
        "--outcsv",
        "--output-csv",
        type=Path,
        help="CSV file to dump segments to",
    )
    parser.add_argument(
        "-f",
        "--format",
        help="if -o is specified, Python format string specifying output path",
    )
    args = parser.parse_args()

    format_str = args.format
    if format_str is None:
        format_str = "{parent}/{stem}/{phone},{start},{end}.npy"

    if args.alignments.suffix == ".csv":
        alignments = iter_alignments_from_csv(args.alignments)
    else:
        alignments = iter_alignments_from_dir(args.alignments)

    frame_transform = lambda frame: int(
        frame * args.upsample // args.downsample
    )

    extract_phones(
        alignments=alignments,
        root=args.root,
        out_root=args.outdir,
        out_csv_path=args.outcsv,
        frame_transform=frame_transform,
        format_str=format_str,
    )
