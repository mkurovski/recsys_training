"""

"""
from itertools import combinations
from typing import Dict, Tuple

import numpy as np
import pandas as pd


from .utils import get_entity_sim


class NearestNeighborRecommender(object):
    """
    Implementation of user-based, neighborhood-based collaborative filtering
    """
    def __init__(self,
                 ratings: pd.DataFrame,
                 users: np.array,
                 items: np.array,
                 k: int = 10,
                 N: int = 5,
                 C: int = 1,
                 metric: str = 'pearson'):
        self.ratings = ratings
        self.users = sorted(users)
        self.items = sorted(items)
        self.k = k
        self.N = N
        self.C = C
        self.metric = metric

        self.user_ratings = {}
        self.user_user_sims = {}

        self.setup()

    def setup(self):
        # rating mapping
        grouped = self.ratings[['user', 'item', 'rating']].groupby('user')
        for user in self.users:
            vals = grouped.get_group(user)[['item', 'rating']].values
            self.user_ratings[user] = dict(zip(vals[:, 0].astype(int),
                                               vals[:, 1].astype(float)))

        # user similarities
        # TODO: Also let define the minimum number of co-rated items to become a relevant neighbor
        user_user = combinations(sorted(self.users), 2)
        for comb in user_user:
            self.user_user_sims[comb] = get_entity_sim(comb[0],
                                                       comb[1],
                                                       self.user_ratings,
                                                       mode=self.metric)

    def get_k_nearest_neighbors(self, user: int) -> Dict[int, float]:
        neighbors = set(self.users)
        neighbors.remove(user)

        nearest_neighbors = dict()
        for neighbor in neighbors:
            sim = self.user_user_sims[tuple(sorted((user, neighbor)))][0]
            if pd.notnull(sim):
                nearest_neighbors[neighbor] = sim

        nearest_neighbors = sorted(nearest_neighbors.items(),
                                   key=lambda kv: kv[1],
                                   reverse=True)

        return nearest_neighbors[:self.k]

    def get_recommendations(self, user: int):
        known_items = list(self.user_ratings[user].keys())
        user_neighbors = self.get_k_nearest_neighbors(user)
        neighborhood_ratings = dict()
        for neighbor, sim in user_neighbors:
            neighbor_ratings = self.user_ratings[neighbor]

            # remove known items
            for known_item in known_items:
                neighbor_ratings.pop(known_item, None)

            # collect neighbor ratings and items
            for item, rating in neighbor_ratings.items():
                add_item = {'sim': sim, 'rating': rating}
                if item not in neighborhood_ratings.keys():
                    neighborhood_ratings[item] = [add_item]
                else:
                    neighborhood_ratings[item].append(add_item)

        rating_preds = self.compute_rating_pred(neighborhood_ratings)
        recs = self.compute_top_n(rating_preds)

        return recs

    @staticmethod
    def compute_rating_pred(neighbor_ratings):
        rating_preds = dict()
        for item, ratings in neighbor_ratings.items():
            sims = np.array([rating['sim'] for rating in ratings])
            ratings = np.array([rating['rating'] for rating in ratings])
            pred_rating = (sims * ratings).sum() / sims.sum()
            count = len(sims)
            rating_preds[item] = {'pred': pred_rating,
                                  'count': count}

        return rating_preds

    def compute_top_n(self, rating_preds):
        rating_preds = {key: val for (key, val) in rating_preds.items()
                        if val['count'] >= self.C}
        sorted_rating_preds = sorted(rating_preds.items(),
                                     key=lambda kv: kv[1]['pred'],
                                     reverse=True)

        return sorted_rating_preds[:self.N]
