# -*- coding: utf-8 -*-
"""

"""
import calendar
import logging
import os

import numpy as np
import pandas as pd

from .utils import min_max_scale


_logger = logging.getLogger(__name__)


genres = [
    'unknown',
    'Action',
    'Adventure',
    'Animation',
    'Children',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Fantasy',
    'Film-Noir',
    'Horror',
    'Musical',
    'Mystery',
    'Romance',
    'Sci-Fi',
    'Thriller',
    'War',
    'Western'
]

# TODO: Generalize initialization into from dataframes and from file
class Dataset(object):
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.load()

    def load(self):
        self.ratings = pd.read_csv(self.filepath,
                                   sep='\t',
                                   header=None,
                                   names=['user', 'item', 'rating', 'timestamp'],
                                   engine='python')
        self.users = sorted(self.ratings['user'].unique())
        self.items = sorted(self.ratings['item'].unique())
        self.n_users = len(self.users)
        self.n_items = len(self.items)
        self.n_ratings = len(self.ratings)

    def rating_split(self, train_size: int = 0.8, seed: int=42):
        # rating split instead of user/item split
        np.random.seed(seed)
        idxs = np.random.choice(self.n_ratings, size=self.n_ratings, replace=False)
        split_point = int(self.n_ratings * train_size)
        train_idxs, test_idxs = idxs[:split_point], idxs[split_point:]
        self.train_ratings = self.ratings.loc[train_idxs]
        self.test_ratings = self.ratings.loc[test_idxs]

    def binarize(self, min_rating: float=4.0):
        """Only keep ratings above threshold as positive implicit feedback"""
        idxs = self.ratings[self.ratings['rating'] >= min_rating].index.values
        self.ratings = self.ratings.loc[idxs, ['user', 'item']]
        self.ratings.reset_index(drop=True, inplace=True)
        self.n_ratings = len(self.ratings)


def preprocess_users(users: pd.DataFrame, zip_digits_to_cut: int=3) -> pd.DataFrame:
    user_age_bounds = {'min': users['age'].min(),
                       'max': users['age'].max()}
    occupations = sorted(users['occupation'].unique())
    user_occupation_map = dict(zip(occupations, range(len(occupations))))
    genders = sorted(users['gender'].unique())
    user_gender_map = dict(zip(genders, range(len(genders))))
    idxs = users[~users['zip'].str.isnumeric()].index
    users.loc[idxs, 'zip'] = '00000'

    users['age'] = users['age'].apply(lambda age: min_max_scale(age, user_age_bounds))
    users['occupation'] = users['occupation'].map(user_occupation_map)
    users['gender'] = users['gender'].map(user_gender_map)
    users['zip'] = users['zip'].apply(lambda val: int(val) // 10 ** zip_digits_to_cut)

    return users


def preprocess_items(items: pd.DataFrame) -> pd.DataFrame:
    idxs = items[items['release'].notnull()].index
    items.loc[idxs, 'release_month'] = items.loc[idxs, 'release'].str.split('-')
    items.loc[idxs, 'release_month'] = \
        items.loc[idxs, 'release_month'].apply(lambda val: val[1])
    items.loc[idxs, 'release_year'] = items.loc[idxs, 'release'].str.split('-')
    items.loc[idxs, 'release_year'] = \
        items.loc[idxs, 'release_year'].apply(lambda val: val[2]).astype(int)

    release_month_map = dict((v, k) for k, v in enumerate(calendar.month_abbr))
    items.loc[idxs, 'release_month'] = items.loc[idxs, 'release_month'].map(
        release_month_map)

    top_month = items['release_month'].value_counts().index[0]
    top_year = items.loc[idxs, 'release_year'].astype(int).describe()['50%']
    # using top month and top year to impute the only missing one
    idx = items[items['release'].isnull()].index
    items.loc[idx, 'release_month'] = top_month
    items.loc[idx, 'release_year'] = top_year

    item_year_bounds = {'min': items['release_year'].min(),
                        'max': items['release_year'].max()}
    items['release_year'] = items['release_year'].apply(
        lambda year: min_max_scale(year, item_year_bounds))
    items.drop(['title', 'release', 'video_release', 'imdb_url'], axis=1, inplace=True)

    return items


def get_user_profiles(ratings: pd.DataFrame, prep_items: pd.DataFrame) -> pd.DataFrame:
    min_rating = 4
    ratings = ratings[ratings.rating >= min_rating]
    ratings.drop(['rating', 'timestamp'], axis=1, inplace=True)
    ratings = ratings.merge(prep_items, on='item', how='left')
    ratings.drop(['item', 'release_month'], axis=1, inplace=True)
    grouped = ratings.groupby('user')
    profiles = grouped.apply(user_profiler).reset_index()
    profiles.rename(columns={'50%': 'median'}, inplace=True)

    return profiles


def user_profiler(group):
    genre_dist = group[genres].mean()
    year_dist = group['release_year'].describe()[['mean', 'std', '50%']]

    return pd.concat((genre_dist, year_dist), axis=0)
