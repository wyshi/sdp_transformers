from typing import Set, Tuple, Dict, Optional, List
import numpy as np
import math


def rr_topk(k: int, truth_y: int, sorted_prior: Dict, eps: float) -> int:
    topk_ys = list(sorted_prior.keys())[:k]
    if truth_y in topk_ys:
        eps_prob = math.exp(eps) / (math.exp(eps) + k - 1)
        output_truth = np.random.choice([0, 1], p=[1 - eps_prob, eps_prob])
        if output_truth:
            return truth_y
        else:
            return np.random.choice(topk_ys.remove(truth_y))
    else:
        return np.random.choice(topk_ys)


def sort_prior(prior):
    keys = prior.keys()
    np.random.shuffle(keys)
    shuffled_prior = {k: prior[k] for k in keys}

    sorted_prior = {
        k: v
        for k, v in sorted(shuffled_prior.items(), key=lambda x: x[1], reverse=True)
    }
    return sorted_prior


def rr_with_prior(sorted_prior: Dict, eps: float) -> int:
    p_cumsum = np.cumsum(sorted_prior.values())  # todo ties broken up arbtritraily
