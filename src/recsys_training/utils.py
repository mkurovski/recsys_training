"""

"""
import logging
import sys

import pandas as pd
import numpy as np
import scipy as sp


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def get_entity_sim(a: int, b: int, entity_ratings, mode: str = 'pearson'):
    """
    Cosine Similarity
    Pearson Correlation
    Adjusted Cosine Similarity
    Jaccard Similarity (intersection over union) - not a good idea as it does not incorporate ratings, e.g.
        even the same users have rated two items, highest Jaccard similarity as evidence for high item similarity,
        their judgement may be very differently on the two items, justifying dissimilarity
    """
    # 1. isolate e.g. users that have rated both items (a and b)
    key_intersection = set(entity_ratings[a].keys()).intersection(entity_ratings[b].keys())
    ratings = np.array([(entity_ratings[a][key], entity_ratings[b][key]) for key in key_intersection])
    n_joint_ratings = len(ratings)

    # Unless provide proper error handling
    # assert(n_joint_ratings > 1)
    if n_joint_ratings > 1:
        # a, b = np.split(ratings, 2, axis=1)
        # 2. apply a similarity computation technique
        if mode == 'pearson':
            sim = np.corrcoef(ratings, rowvar=False)[0, 1]
        elif mode == 'cosine':
            nom = ratings[:, 0].dot(ratings[:, 1])
            denom = np.linalg.norm(ratings[:, 0]) * np.linalg.norm(ratings[:, 1])
            sim = nom / denom
        elif mode == 'euclidean':
            sim = normalized_euclidean_sim(ratings[:, 0], ratings[:, 1])
        elif mode == 'adj_cosine':
            sim = None
        else:
            raise ValueError(f"Value {mode} for argument 'mode' not supported.")
    else:
        sim = None

    return sim, n_joint_ratings


def normalized_euclidean_sim(a, b):
    # scale to unit vectors
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)

    dist = np.linalg.norm(a_norm - b_norm)
    sim = 2 - dist - 1
    return sim


def min_max_scale(val, bounds):
    min_max_range = bounds['max']-bounds['min']
    return (val-bounds['min'])/min_max_range


def sigmoid(x):
    return 1/(1+np.exp(-x))


def df_to_coo(df, n_users, n_items):
    coo = sp.sparse.coo_matrix(([1]*len(df), (df.user.values-1, df.item.values-1)),
                               shape=(n_users, n_items), dtype=np.int32)
    return coo


def coo_to_df(coo):
    mat = np.concatenate((coo.row.reshape(-1, 1)+1,
                          coo.col.reshape(-1, 1)+1),
                         axis=1)
    return pd.DataFrame(mat, columns=['user', 'item'])
