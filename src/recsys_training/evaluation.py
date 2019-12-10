from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def compute_mae(test_ratings: pd.DataFrame, recommender) -> Tuple[float, float]:
    pred = test_ratings.apply(lambda row:
                              recommender.get_prediction(row['user'], row['item']),
                              axis=1)

    pred = pred.apply(lambda val: list(val.values())[0]['pred'])
    notnulls = pred.notnull()
    mae = np.mean(np.abs(test_ratings.rating[notnulls] - pred[notnulls]))
    coverage = notnulls.sum()/len(test_ratings)

    return {'mae': mae, 'coverage': coverage}


def retrieval_score(test_ratings: pd.DataFrame,
                    recommender,
                    metric: str='mrr',
                    min_rating: int=4) -> float:
    """
    Mean Average Precision / Mean Reciprocal Rank of first relevant item @ N
    """
    N = recommender.N
    user_scores = []
    relevant_items = get_relevant_items(test_ratings, min_rating)

    for user in recommender.users:
        if user in relevant_items.keys():
            predicted_items = dict(recommender.get_recommendations(user))
            predicted_items = [item for item, pred in predicted_items.items()]
            if metric == 'map':
                true_positives = np.intersect1d(relevant_items[user],
                                                predicted_items)
                score = len(true_positives) / N
            elif metric == 'mrr':
                score = np.mean([reciprocal_rank(item, predicted_items)
                                 for item in relevant_items[user]])
                # import pdb; pdb.set_trace()
            else:
                raise ValueError(f"Unknown value {metric} for Argument `metric`")

            user_scores.append(score)

    return np.mean(user_scores)


def reciprocal_rank(item: int, ranking: List[int]) -> float:
    rr = 0
    if item in ranking:
        rr = 1/(ranking.index(item)+1)

    return rr


def get_relevant_items(test_ratings: pd.DataFrame, min_rating: int=4) -> Dict[int, List[int]]:
    """
    returns {user: [items]} as relevant items per user
    """
    relevant_items = test_ratings[test_ratings.rating >= min_rating][['user', 'item']]
    relevant_items = relevant_items.groupby('user')
    relevant_items = {user: relevant_items.get_group(user)['item'].values
                      for user in relevant_items.groups.keys()}

    return relevant_items
