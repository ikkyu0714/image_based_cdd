from __future__ import annotations

import itertools

import numpy as np


def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = np.dot(
        vector_a,
        vector_b,
    ) / (norm_a * norm_b)

    return float(similarity)


def cosine_distance(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
) -> float:
    return 1.0 - cosine_similarity(
        vector_a,
        vector_b,
    )


def mean_pairwise_euclidean_distance(
    features: np.ndarray,
) -> float:
    if len(features) < 2:
        return 0.0

    distances: list[float] = []

    for vector_a, vector_b in itertools.combinations(
        features,
        2,
    ):
        distance = np.linalg.norm(vector_a - vector_b)

        distances.append(float(distance))

    return float(np.mean(distances))
