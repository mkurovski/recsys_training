# -*- coding: utf-8 -*-
"""

"""
import logging
import os

import numpy as np
import pandas as pd


_logger = logging.getLogger(__name__)


class Dataset(object):
    def __init__(self, filepath: str, seed: int = 42):
        self.filepath = filepath
        self.seed = seed
        self.load()

    def load(self):
        self.ratings = pd.read_csv(self.filepath,
                                   sep='\t',
                                   header=None,
                                   names=['user', 'item', 'rating', 'timestamp'],
                                   engine='python')
        self.users = sorted(self.ratings['user'].unique())
        self.items = sorted(self.ratings['item'].unique())
        self.n_user = len(self.users)
        self.n_items = len(self.items)
        self.n_ratings = len(self.ratings)

    def rating_split(self, train_size: int = 0.8):
        # rating split instead of user/item split
        np.random.seed(self.seed)
        idxs = np.random.choice(self.n_ratings, size=self.n_ratings, replace=False)
        split_point = int(self.n_ratings * train_size)
        train_idxs, test_idxs = idxs[:split_point], idxs[split_point:]
        self.train_ratings = self.ratings.loc[train_idxs]
        self.test_ratings = self.ratings.loc[test_idxs]
