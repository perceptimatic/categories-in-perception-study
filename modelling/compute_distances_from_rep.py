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

def skld(x, y, eps):
    x[x == 0] = eps
    y[y == 0] = eps
    kldxy = scipy.stats.entropy(x, y)
    kldyx = scipy.stats.entropy(y, x)
    return 0.5*(kldyx + kldxy)

def skld_cdist(rep_a, rep_b, eps=0.001):
    return scipy.spatial.distance.cdist(rep_a, rep_b,
                                        metric=lambda x, y: skld(x, y, eps)).astype(np.float32)

def cosine_dist(rep_a, rep_b):
    return scipy.spatial.distance.cdist(rep_a, rep_b,
                                        metric="cosine").astype(np.float32)

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

def get_rep(item_name, rep_location):
    if isinstance(rep_location, str):
        return np.load(get_filename(item_name, rep_location))
    elif isinstance(rep_location, pd.DataFrame):
        return rep_location.loc[rep_location['filename'] == item_name,:].drop('filename', axis=1).to_numpy()
        

def get_pair_name(name_a, name_b):
    return tuple(sorted((name_a, name_b)))

def get_distance(name_a, name_b, distance, pooling, cached_distances, rep_path):
    if os.path.isdir(rep_path):
        rep_location = rep_path
    elif os.path.isfile(rep_path):
        rep_location = pd.read_csv(rep_path)
    else:
        raise RuntimeError()
        
    pair_name = get_pair_name(name_a, name_b)
    try:
        d = cached_distances[pair_name]
    except KeyError:
        rep_a = get_rep(name_a, rep_location)
        rep_b = get_rep(name_b, rep_location)
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
    args = parser.parse_args()

    triplets = pd.read_csv(args.triplet_list_file)
    distances_tgt, distances_oth = get_distances(triplets,
                              args.path_to_data,
                              {"kl": skld_cdist, "cosine": cosine_dist}[args.distance_fn],
                              {"dtw": dtw, "window-5": lambda x,y,d: window(x,y,d,5),
                               "window-3": lambda x,y,d: window(x,y,d,3)}[args.pooling])
    triplets[args.column_prefix + '_distance_tgt'] = distances_tgt
    triplets[args.column_prefix + '_distance_oth'] = distances_oth
    triplets.to_csv(args.out_fn, index=False)
