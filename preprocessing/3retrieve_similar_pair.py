import argparse
import os

import numpy as np
import pandas as pd
import scipy.spatial
from tqdm import tqdm

SUBSET = ['val', 'train', 'test']


def vector_distance(vec1, vec2, method = "l2", l2_normalize = True):
    """Computes the distance between 2 vectors
    Inspired by https://github.com/Azure/ImageSimilarityUsingCntk=
    Args:
        vec1: First vector between which the distance will be computed
        vec2: Second vector
        method: Type of distance to be computed, e.g. "l1" or "l2"
        l2_normalize: Flag indicating whether the vectors should be normalized
        to be of unit length before the distance between them is computed
    Returns: Distance between the 2 input vectors
    """
    # Pre-processing
    if l2_normalize:
        vec1 = vec1 / np.linalg.norm(vec1, 2)
        vec2 = vec2 / np.linalg.norm(vec2, 2)

    # Distance computation
    vecDiff = vec1 - vec2
    method = method.lower()
    if method == "l1":
        dist = sum(abs(vecDiff))
    elif method == "l2":
        dist = np.linalg.norm(vecDiff, 2)
    elif method == "normalizedl2":
        a = vec1 / np.linalg.norm(vec1, 2)
        b = vec2 / np.linalg.norm(vec2, 2)
        dist = np.linalg.norm(a - b, 2)
    elif method == "cosine":
        dist = scipy.spatial.distance.cosine(vec1, vec2)
    elif method == "correlation":
        dist = scipy.spatial.distance.correlation(vec1, vec2)
    elif method == "chisquared":
        dist = scipy.chiSquared(vec1, vec2)
    elif method == "normalizedchisquared":
        a = vec1 / sum(vec1)
        b = vec2 / sum(vec2)
        dist = scipy.chiSquared(a, b)
    elif method == "hamming":
        dist = scipy.spatial.distance.hamming(vec1 > 0, vec2 > 0)
    else:
        raise Exception("Distance method unknown: " + method)
    return dist


def get_simi(args, topk=10):
    npy_file = {}
    # load all data to memory
    split_file = None
    for subset in SUBSET:
        if "test" == subset:
            split_file = pd.read_csv(os.path.join(args.root, args.test_split))
        elif "train" == subset:
            split_file = pd.read_csv(os.path.join(args.root, args.train_split))
        elif "val" == subset:
            split_file = pd.read_csv(os.path.join(args.root, args.val_split))

        for filename in split_file['filename']:
            filePath = os.path.join(args.root, os.path.join(args.feature, filename[:-3] + 'npy'))
            npy_file[filename] = np.load(filePath).reshape(-1)

    # all similar from train set
    train_lists = pd.read_csv(os.path.join(args.root, args.train_split))['filename']

    # prepare store top10 similar information by cosion distance
    simi_info = {"filename": [], 'split': [], 'method': []}
    for i in range(topk):
        simi_info["similar_{:d}".format(i)] = []
        simi_info["distance_{:d}".format(i)] = []

    for subset in SUBSET:
        print("========>> precess {} set".format(subset))

        if "test" == subset:
            split_file = pd.read_csv(os.path.join(args.root, args.test_split))
        elif "train" == subset:
            split_file = pd.read_csv(os.path.join(args.root, args.train_split))
        elif "val" == subset:
            split_file = pd.read_csv(os.path.join(args.root, args.val_split))

        for target in tqdm(split_file['filename']):
            this_simi = []
            for pair_filename in train_lists:
                if target == pair_filename:
                    continue

                dist = vector_distance(npy_file[target], npy_file[pair_filename], args.method)
                this_simi.append({"pair_filename": pair_filename, "similar_distance": dist})

            # store this image's top10 similar information
            this_simi = sorted(this_simi, key=lambda x: x["similar_distance"])
            simi_info['filename'].append(target)
            simi_info['split'].append(subset)
            simi_info['method'].append(args.method)
            for i, this_simi in enumerate(this_simi[:topk]):
                simi_info["similar_{:d}".format(i)].append(this_simi["pair_filename"])
                simi_info["distance_{:d}".format(i)].append(this_simi["similar_distance"])

    csv_filename = os.path.join(args.output, 'pair_list.csv')
    csv_df = pd.DataFrame(simi_info)
    csv_df.to_csv(csv_filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--root', type=str, default="data",
                        help="the begin directory of all")
    parser.add_argument('-s', '--test_split', type=str, default="test_split.csv",
                        help="test split file")
    parser.add_argument('-a', '--train_split', type=str, default="train_split.csv",
                        help="train split file")
    parser.add_argument('-b', '--val_split', type=str, default="val_split.csv",
                        help="val split file")
    parser.add_argument('-f', '--feature', type=str, default="feature",
                        help="feature dir")
    parser.add_argument('-o', '--output', type=str, default="",
                        help="output file save dir")
    parser.add_argument('-m', '--method', type=str, default="cosine",
                        help="similarity method")

    args = parser.parse_args()
    print(args)

    get_simi(args)