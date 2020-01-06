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


# TODO: Remove min_rating logic from here (should be done before on data through binarize)
def retrieval_score(test_ratings: pd.DataFrame,
                    recommender,
                    remove_known_pos: bool = False,
                    metric: str = 'mrr') -> float:
    """
    Mean Average Precision / Mean Reciprocal Rank of first relevant item @ N
    """
    N = recommender.N
    user_scores = []
    relevant_items = get_relevant_items(test_ratings)

    for user in recommender.users:
        if user in relevant_items.keys():
            predicted_items = recommender.get_recommendations(user, remove_known_pos)
            predicted_items = [item for item, _ in predicted_items]
            if metric == 'map':
                true_positives = np.intersect1d(relevant_items[user],
                                                predicted_items)
                score = len(true_positives) / N
            elif metric == 'mrr':
                score = np.mean([reciprocal_rank(item, predicted_items)
                                 for item in relevant_items[user]])
            else:
                raise ValueError(f"Unknown value {metric} for Argument `metric`")

            user_scores.append(score)

    return np.mean(user_scores)


def reciprocal_rank(item: int, ranking: List[int]) -> float:
    rr = 0
    if item in ranking:
        rr = 1/(ranking.index(item)+1)

    return rr


def get_relevant_items(test_ratings: pd.DataFrame) -> Dict[int, List[int]]:
    """
    returns {user: [items]} as relevant items per user
    """
    relevant_items = test_ratings[['user', 'item']]
    relevant_items = relevant_items.groupby('user')
    relevant_items = {user: relevant_items.get_group(user)['item'].values
                      for user in relevant_items.groups.keys()}

    return relevant_items
