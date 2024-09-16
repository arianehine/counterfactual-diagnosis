from functools import lru_cache

import numpy as np
from scipy.stats import binom_test

from constants import NETWORKS_FILE
from utils import load_from_json


@lru_cache(maxsize=1)
def load_networks(datapath, filename=NETWORKS_FILE):
    return load_from_json(datapath / filename)


def get_doctor_differential(li):
    return [val["concept"]["id"] for val in li if val["concept"]["id"] is not None]
#
# Using produce_differentials, the system gathers doctors’ diagnoses and compares them with the true diagnosis using doctor_top_ns to see whether they were accurate. It can also calculate the average number of diseases considered by each doctor using mean_list.
def produce_differentials(card):
    return dict(
        [
            [val["user"]["id"], get_doctor_differential(val["doctor_diseases"])]
            for val in card["outcomes"]
            if "doctor_diseases" in val
        ]
    )


def doctor_top_ns(card, true_disease):
    differentials = produce_differentials(card)
    return [
        [key, len(val), 1 if true_disease in val else 0]
        for key, val in differentials.items()
    ]


def mean_list(li):
    if li == []:
        return "none"
    else:
        return np.mean(li)
#
# Finally, the bintest function is used to determine whether the observed results are statistically significant. This helps assess whether the observed diagnostic accuracy is meaningful or just due to chance.
def bintest(x, y, comf_thresh):
    if comf_thresh == 0:
        return sum(x) >= sum(y)
    return (
        binom_test(sum(x), n=len(x), p=sum(y) / len(y), alternative="greater")
        < comf_thresh
    )
