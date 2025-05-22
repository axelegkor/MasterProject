import torch
import numpy as np
from classes.antibody import Antibody
from classes.antigen import Antigen
from typing import List, Tuple


def whiten_embedding(
    embedding: np.ndarray, mu: np.ndarray, w: np.ndarray
) -> np.ndarray:
    return np.dot(embedding - mu, w)


def classify_antigen(
    voting_method: str,
    antigen: Antigen,
    antibodies: List[Antibody],
    whitening: bool,
    mu: np.ndarray,
    w: np.ndarray,
    forced_coverage: bool,
) -> int:
    if whitening and mu is not None and w is not None:
        antigen.embedding = whiten_embedding(antigen.embedding, mu, w)
    votes = {}
    for antibody in antibodies:
        if antibody.is_recognized(antigen):
            votes[antibody.label] = votes.get(antibody.label, 0) + antibody.accuracy
    if not votes:
        if forced_coverage:
            closest = (None, 0)
            for antibody in antibodies:
                if closest[0] is None or antibody.scaled_distance(antigen) < closest[1]:
                    closest = (antibody, antibody.scaled_distance(antigen))
            return closest[0].label
        else:
            return -1
    if voting_method == "binary":
        return max(votes, key=votes.get)
    elif voting_method == "continuous":
        total_votes = sum(votes.values())
        weighted_sum = sum(label * (votes[label] / total_votes) for label in votes)
        return int(np.round(weighted_sum))
    else:
        raise ValueError(f"Unknown voting method: {voting_method}")
