"""

"""
from collections import OrderedDict
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .evaluation import retrieval_score
from .utils import get_entity_sim, sigmoid


# TODO: Implement biases
# TODO: Implement Adaptive Learning Rate, e.g. Adam Optimizer
# TODO: Transform from MBGD to pure SGD
class BPRRecommender(object):
    def __init__(self, ratings: pd.DataFrame, users: np.array, items: np.array,
                 k: int, N: int, seed: int = 42):
        self.ratings = ratings
        self.users = sorted(users)
        self.items = sorted(items)
        self.m = len(self.users)
        self.n = len(self.items)
        self.k = k
        self.N = N
        self.user_pos_items = {}
        self.user_neg_items = {}
        self.user_factors = None
        self.item_factors = None
        self._setup(seed)

    def _setup(self, seed: int = 42):
        random_state = np.random.RandomState(seed)

        # Latent Factor initialization according to LightFM
        self.user_factors = (random_state.rand(self.m, self.k) - 0.5) / self.k
        self.item_factors = (random_state.rand(self.n, self.k) - 0.5) / self.k

        # Shuffle Ratings to break user-item contingency in SGD updates
        # and thus avoid slow convergence
        self.ratings = self.ratings.sample(frac=1, random_state=seed)

        self._init_user_ratings()

    def _init_user_items(self):
        grouped = self.ratings[['user', 'item']].groupby('user')
        groups = grouped.groups.keys()
        for user in self.users:
            pos_items = []
            if user in groups:
                pos_items = grouped.get_group(user).item.values
            neg_items = np.setdiff1d(self.items, pos_items)
            self.user_pos_items[user] = pos_items
            self.user_neg_items[user] = neg_items

    # TODO: Implement weight decay regularization
    # TODO: Shorten
    def train(self, epochs: int, batch_size: int, learning_rate: float) -> List[float]:
        num_batches = int(np.ceil(len(self.ratings) / batch_size))
        correct_trace = []
        auc_trace = []

        for epoch in range(epochs):
            for idx in range(num_batches):
                minibatch = self.ratings.iloc[idx * batch_size:(idx + 1) * batch_size]
                users, pos_items = np.hsplit(minibatch.values, 2)
                users = users.flatten()
                pos_items = pos_items.flatten()
                neg_items = self.negative_sampling(users, pos_items)
                # import pdb; pdb.set_trace()
                # deduct 1 as user ids are 1-indexed, but array is 0-indexed
                user_embeds = self.user_factors[users - 1]
                pos_item_embeds = self.item_factors[pos_items - 1]
                neg_item_embeds = self.item_factors[neg_items - 1]
                # pdb.set_trace()
                user_grads, pos_item_grads, neg_item_grads = \
                    self.compute_gradients(user_embeds, pos_item_embeds, neg_item_embeds)
                # pdb.set_trace()
                # update
                # update fails for multiple same users or items within a batch
                # TODO: Correct accordingly here and in Multi-Channel BPR Repo
                # also try running only as shochastic updates
                # import pdb; pdb.set_trace()
                self.user_factors[users - 1] -= learning_rate * user_grads
                self.item_factors[pos_items - 1] -= learning_rate * pos_item_grads
                self.item_factors[neg_items - 1] -= learning_rate * neg_item_grads
                # pdb.set_trace()
                if not idx % 500:
                    pos_interact = np.sum(user_embeds * pos_item_embeds, axis=1)
                    neg_interact = np.sum(user_embeds * neg_item_embeds, axis=1)
                    x_uij = pos_interact - neg_interact
                    correct_rankings = (x_uij > 0).sum() / x_uij.shape[0]
                    auc = sigmoid(x_uij).mean()
                    correct_trace.append(correct_rankings)
                    auc_trace.append(auc)
                    print(f"Correct Rankings: {correct_rankings:.1%}, AUC: {auc:.3f}")

        return correct_trace, auc_trace

    def negative_sampling(self, users: np.array, pos_items: np.array) -> np.array:
        """
        Return the item ids for negative samples
        """
        # TODO: Allow for popularity-based pdf for choosing negative item
        neg_items = []
        for user in users:
            neg_item = np.random.choice(self.user_neg_items[user])
            neg_items.append(neg_item)

        return np.array(neg_items)

    @staticmethod
    def compute_gradients(user_embeds: np.array, pos_item_embeds: np.array,
                          neg_item_embeds: np.array) -> Tuple[np.array]:
        """
        LightFM

        loss = (1.0 - sigmoid(positive_prediction - negative_prediction))

        update:
        pos_item: -loss * user_component
        neg_item: loss * user_component
        user: loss * (negative_item_component - positive_item_component)
        """
        # TODO: also test for stochastic case the shape to be 2-D
        # TODO: pred = pos_pred - neg_pred (renaming)
        x_ui = np.sum(user_embeds * pos_item_embeds, axis=1)
        x_uj = np.sum(user_embeds * neg_item_embeds, axis=1)
        x_uij = x_ui - x_uj

        generic_grad = (-np.exp(-x_uij) * sigmoid(x_uij)).reshape(-1, 1)
        test_generic_grad = (1.0 - sigmoid(x_ui - x_uj)).reshape(-1, 1)

        user_grad = generic_grad * (pos_item_embeds - neg_item_embeds)
        pos_item_grad = generic_grad * user_embeds
        neg_item_grad = generic_grad * (-user_embeds)

        test_user_grad = test_generic_grad * (neg_item_embeds - pos_item_embeds)
        test_pos_item_grad = -test_generic_grad * user_embeds
        test_neg_item_grad = test_generic_grad * user_embeds

        # import pdb; pdb.set_trace()

        return user_grad, pos_item_grad, neg_item_grad

    def get_recommendations(self, user: int) -> List[Tuple[int, Dict[str, float]]]:
        known_items = list(self.user_ratings[user])
        predictions = self.get_prediction(user)
        recommendations = []
        for item, pred in predictions.items():
            if item not in known_items:
                add_item = (item, pred)
                recommendations.append(add_item)
            if len(recommendations) == self.N:
                break

        return recommendations

    def get_prediction(self, user: int, items: np.array = None) -> Dict[int, Dict[str, float]]:
        if items is None:
            items = self.items
        if type(items) == np.int64:
            items = [items]
        items = np.array(items)

        user_embed = self.user_factors[user - 1].reshape(1, -1)
        item_embeds = self.item_factors[items - 1].reshape(len(items), -1)

        # use array-broadcasting
        preds = np.sum(user_embed * item_embeds, axis=1)
        sorting = np.argsort(preds)[::-1]
        preds = {item: {'pred': pred} for item, pred in
                 zip(items[sorting], preds[sorting])}

        return preds


class MFRecommender(object):
    def __init__(self, ratings: pd.DataFrame, users: np.array, items: np.array,
                 k: int, N: int, seed: int = 42):
        self.ratings = ratings
        self.users = sorted(users)
        self.items = sorted(items)
        self.m = len(self.users)
        self.n = len(self.items)
        self.k = k
        self.N = N
        self.user_ratings = {}
        self.setup(seed)

    def setup(self, seed: int = 42):
        np.random.seed(seed)
        self.user_factors = np.random.normal(0, 1, (self.m, self.k))
        self.item_factors = np.random.normal(0, 1, (self.n, self.k))
        self.ratings = self.ratings.sample(frac=1, random_state=seed)

        grouped = self.ratings[['user', 'item', 'rating']].groupby('user')
        for user in self.users:
            vals = grouped.get_group(user)[['item', 'rating']].values
            self.user_ratings[user] = dict(zip(vals[:, 0].astype(int),
                                               vals[:, 1].astype(float)))

    # TODO: Implement weight decay regularization
    def train(self, epochs: int, batch_size: int, learning_rate: float) -> List[float]:
        num_batches = int(np.ceil(len(self.ratings) / batch_size))
        rmse_trace = []
        for epoch in range(epochs):
            for idx in range(num_batches):
                minibatch = self.ratings.iloc[idx * batch_size:(idx + 1) * batch_size][
                    ['user', 'item', 'rating']]
                # deduct 1 as user ids are 1-indexed, but array is 0-indexed
                user_embeds = self.user_factors[minibatch['user'].values - 1]
                item_embeds = self.item_factors[minibatch['item'].values - 1]

                user_grads, item_grads = self.compute_gradients(
                        minibatch['rating'].values,
                        user_embeds,
                        item_embeds)

                self.user_factors[
                    minibatch['user'].values - 1] -= learning_rate * user_grads
                self.item_factors[
                    minibatch['item'].values - 1] -= learning_rate * item_grads

                if not idx % 50:
                    rmse = (
                        self.rmse(minibatch['rating'].values, user_embeds, item_embeds))
                    rmse_trace.append(rmse)
                    print(f"{rmse:.3f}")

        return rmse_trace

    def get_recommendations(self, user: int) -> List[Tuple[int, Dict[str, float]]]:
        known_items = list(self.user_ratings[user].keys())
        predictions = self.get_prediction(user)
        recommendations = []
        for item, pred in predictions.items():
            if item not in known_items:
                add_item = (item, pred)
                recommendations.append(add_item)
            if len(recommendations) == self.N:
                break

        return recommendations

    def get_prediction(self, user: int, items: np.array = None) -> Dict[int, Dict[str, float]]:
        if items is None:
            items = self.items
        if type(items) == np.int64:
            items = [items]
        items = np.array(items)

        user_embed = self.user_factors[user - 1].reshape(1, -1)
        item_embeds = self.item_factors[items - 1].reshape(len(items), -1)

        # use array-broadcasting
        preds = np.sum(user_embed * item_embeds, axis=1)
        sorting = np.argsort(preds)[::-1]
        preds = {item: {'pred': pred} for item, pred in
                 zip(items[sorting], preds[sorting])}

        return preds

    @staticmethod
    def compute_gradients(rating: float, u: np.array, v: np.array) -> Tuple[np.array]:
        # TODO: also test for stochastic case the shape to be 2-D
        pred = np.sum(u * v, axis=1)
        error = (rating - pred).reshape(-1, 1)

        u_grad = -2 * error * v
        v_grad = -2 * error * u

        return u_grad, v_grad

    @staticmethod
    def rmse(rating, u, v) -> float:
        pred = np.sum(u * v, axis=1)
        error = rating - pred
        rmse = np.sqrt(np.mean(error ** 2))
        return rmse


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

    # TODO: Instead of List of Tuples return List of Tuple with int user and Dict with sim and count
    def get_k_nearest_neighbors(self, user: int) -> List[Tuple[int, float]]:
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

    # TODO: Better return List of Dicts to stay consistent with previous
    def get_recommendations(self, user: int) -> List[Tuple[int, Dict[str, float]]]:
        """
        returns
        """
        known_items = list(self.user_ratings[user].keys())
        user_neighbors = self.get_k_nearest_neighbors(user)
        neighborhood_ratings = dict()
        for neighbor, sim in user_neighbors:
            neighbor_ratings = self.user_ratings[neighbor].copy()

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

    def get_prediction(self, user: int, item: int) -> Dict[int, Dict[str, float]]:
        """
        returns {item: {prediction, count}} for user-item combination
        """
        neighbors = self.get_k_nearest_neighbors(user)
        neighborhood_ratings = {item: list()}
        for neighbor, sim in neighbors:
            if item in self.user_ratings[neighbor].keys():
                rating = self.user_ratings[neighbor][item]
                add_item = {'sim': sim, 'rating': rating}
                neighborhood_ratings[item].append(add_item)

        pred = self.compute_rating_pred(neighborhood_ratings)

        return pred

    # TODO: Assumes that neighborhood_ratings is nonempty, provide default answer
    @staticmethod
    def compute_rating_pred(neighborhood_ratings):
        rating_preds = dict()
        for item, ratings in neighborhood_ratings.items():
            if len(ratings) > 0:
                sims = np.array([rating['sim'] for rating in ratings])
                ratings = np.array([rating['rating'] for rating in ratings])
                pred_rating = (sims * ratings).sum() / sims.sum()
                count = len(sims)
                rating_preds[item] = {'pred': pred_rating,
                                      'count': count}
            else:
                rating_preds[item] = {'pred': None, 'count': 0}

        return rating_preds

    def compute_top_n(self, rating_preds):
        rating_preds = {key: val for (key, val) in rating_preds.items()
                        if val['count'] >= self.C}
        # try to break ties of `pred` by descending `count`
        # assuming more ratings mean higher confidence in the prediction
        sorted_rating_preds = sorted(rating_preds.items(),
                                     key=lambda kv: (kv[1]['pred'], kv[1]['count']),
                                     reverse=True)

        return OrderedDict(sorted_rating_preds[:self.N])


class PopularityRecommender(object):
    """
    Implementation of user-based, neighborhood-based collaborative filtering
    * make it time-based to account for seasonal titles
    """
    def __init__(self,
                 ratings: pd.DataFrame,
                 users: np.array,
                 items: np.array,
                 N: int):
        self.ratings = ratings
        self.users = sorted(users)
        self.items = sorted(items)
        self.N = N

        self.setup()

    def setup(self):
        self.item_popularity = self.ratings['item'].value_counts()
        self.item_order = self.item_popularity.index.values

    def get_recommendations(self, user: int=None) -> List[int]:
        recs = dict(zip(self.item_order[:self.N], [None]*self.N))
        return recs


