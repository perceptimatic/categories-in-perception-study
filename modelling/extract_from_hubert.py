#!/usr/bin/env python3

"""
Created by Juliette MILLET september 2022
script to extract from all the layers of hubert model

Modified 2025 Ewan Dunbar
"""

from hubert_pretrained_extraction import HubertFeatureReader
import os
import numpy as np


def main(
    input_directory,
    csv_file,
    ckpt_path,
    layer,
    feat_dir,
    max_chunk,
    filename_column="#file_extract",
):
    print(ckpt_path)
    if not os.path.exists(feat_dir):
        os.makedirs(feat_dir)
    reader = HubertFeatureReader(ckpt_path, layer, max_chunk)
    if csv_file != "":  # then it means it is a csv file
        f = open(csv_file, "r")
        ind = f.readline().replace("\n", "").split(",")
        for line in f:
            new_line = line.replace("\n", "").split(",")
            fili = new_line[ind.index(filename_column)]
            feat = reader.get_feats(path=os.path.join(input_directory, fili))
            np.save(os.path.join(feat_dir, fili.replace(".wav", ".npy")), feat)
    else:
        for file in os.listdir(input_directory):
            if not file.endswith(".wav"):
                continue
            feat = reader.get_feats(path=os.path.join(input_directory, file))
            np.save(os.path.join(feat_dir, file.replace(".wav", ".npy")), feat)
    print("Done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("input_directory")
    parser.add_argument("csv_file")
    parser.add_argument("ckpt_path")
    parser.add_argument("feat_dir")
    parser.add_argument("layers", type=str, help="layers to extract")
    parser.add_argument("--max_chunk", type=int, default=1600000)
    parser.add_argument("--filename_column", type=str, default="#file_extract")
    args = parser.parse_args()
    # logger.info(args)

    for layer in args.layers.split(","):
        main(
            args.input_directory,
            args.csv_file,
            args.ckpt_path,
            layer,
            args.feat_dir + "/" + layer,
            args.max_chunk,
            args.filename_column,
        )
