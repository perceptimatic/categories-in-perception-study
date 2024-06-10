#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created march 2021
    by Juliette MILLET
    Modified June 2024
    Ewan Dunbar
    script to compute distance from representation
"""
import os
import numpy as np
import argparse
import scipy
import pandas as pd
from dtw import _dtw

def skld(rep_a, rep_b, eps=0.001):
    rep_a[rep_a == 0] = eps
    rep_b[rep_b == 0] = eps
    kldab = scipy.stats.entropy(rep_a, rep_b, axis=0, keepdims=True)
    kldba = scipy.stats.entropy(rep_b, rep_b, axis=0, keepdims=True)
    return 0.5*(kldab + kldba)

def cosine_dist(rep_a, rep_b):
    return scipy.spatial.distance.cdist(rep_a, rep_b,
                                        metric="cosine")

def dtw(rep_a, rep_b, distance):
    return _dtw(rep_a.shape[0],
                rep_b.shape[0],
                distance(rep_a, rep_b),
                True)

def window(rep_a, rep_b, distance, k):
    if k <= 0:
        raise RuntimeError("k must be at least 1")
    left_width, right_width = k//2, k - k//2 - 1
    reps = (rep_a, rep_b)
    mids = [x.shape[0]//2 for x in reps]
    lefts = [x - left_width for x in mids]
    rights = [x + right_width for x in mids]
    windows = [reps[i][lefts[i]:rights[i],:] for i in (0,1)]
    pooled_a, pooled_b = [x.mean(axis=0, keepdims=True) for x in windows]
    return distance(pooled_a, pooled_b)[0,0]

def get_filename(item_name, path):
    return os.path.join(path, item_name + ".npy")

def get_rep(filename):
    return np.load(filename)

def get_pair_name(name_a, name_b):
    return tuple(sorted((name_a, name_b)))

def get_distance(name_a, name_b, distance, pooling, cached_distances, rep_path):
    pair_name = get_pair_name(name_a, name_b)
    try:
        d = cached_distances[pair_name]
    except KeyError:
        rep_a = get_rep(get_filename(name_a, rep_path))
        rep_b = get_rep(get_filename(name_b, rep_path))
        d = pooling(rep_a, rep_b, distance)
        cached_distances[pair_name] = d
    return d

def get_distances(triplet_df, rep_path, distance, pooling):
    distances_tgt = []
    distances_oth = []
    deltas = []
    cached_distances = {}
    for i, row in triplet_df.iterrows():
        tgt_item = row['TGT_item']
        oth_item = row['OTH_item']
        x_item = row['X_item']
        distances_tgt.append(get_distance(tgt_item, x_item, distance, pooling, cached_distances, rep_path))
        distances_oth.append(get_distance(oth_item, x_item, distance, pooling, cached_distances, rep_path))
        deltas.append(distances_oth[i] - distances_tgt[i])
    return distances_tgt, distances_oth

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='script to compute distances')
    parser.add_argument('path_to_data', type=str, help='path to representations')
    parser.add_argument('triplet_list_file', type=str, help='files with a list of triplet')
    parser.add_argument('out_fn', type=str, help='out_fn')
    parser.add_argument('distance_fn', type=str)
    parser.add_argument('pooling', type=str)
    parser.add_argument('column_prefix', type=str)
    parser.add_argument('n_threads', type=int)
    args = parser.parse_args()

    triplets = pd.read_csv(args.triplet_list_file)
    distances_tgt, distances_oth = get_distances(triplets,
                              args.path_to_data,
                              {"kl": skld, "cosine": cosine_dist}[args.distance_fn],
                              {"dtw": dtw, "window-5": lambda x,y,d: window(x,y,d,5),
                               "window-3": lambda x,y,d: window(x,y,d,3)}[args.pooling])
    triplets[args.column_prefix + '_distance_tgt'] = distances_tgt
    triplets[args.column_prefix + '_distance_oth'] = distances_oth
    triplets.to_csv(args.out_fn, index=False)
