from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest

FeatureDict = dict[str, list[np.ndarray]]


def flatten_features(
    feature_dict: FeatureDict,
) -> np.ndarray:
    features: list[np.ndarray] = []

    for vectors in feature_dict.values():
        features.extend(vectors)

    if not features:
        raise ValueError("No feature vectors were found.")

    return np.stack(features)


def average(
    feature_dict: FeatureDict,
) -> np.ndarray:
    features = flatten_features(feature_dict)

    return np.mean(
        features,
        axis=0,
    )


def centroid_of_centroids(
    feature_dict: FeatureDict,
) -> np.ndarray:
    centroids: list[np.ndarray] = []

    for vectors in feature_dict.values():
        if not vectors:
            continue

        group = np.stack(vectors)

        centroid = np.mean(
            group,
            axis=0,
        )

        centroids.append(centroid)

    if not centroids:
        raise ValueError("No feature groups were found.")

    return np.mean(
        np.stack(centroids),
        axis=0,
    )


def kmeans(
    feature_dict: FeatureDict,
    n_clusters: int,
    random_state: int,
) -> np.ndarray:
    features = flatten_features(feature_dict)

    if len(features) < n_clusters:
        raise ValueError("The number of feature vectors must be " "greater than or equal to n_clusters.")

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto",
    )

    labels = model.fit_predict(features)

    return _cluster_weighted_average(
        features=features,
        labels=labels,
        ignore_noise=False,
    )


def dbscan(
    feature_dict: FeatureDict,
    eps: float,
    min_samples: int,
) -> np.ndarray:
    features = flatten_features(feature_dict)

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    )

    labels = model.fit_predict(features)

    valid_mask = labels != -1

    if not np.any(valid_mask):
        print("[Warning] DBSCAN classified all " "samples as noise. Falling back to average.")

        return np.mean(
            features,
            axis=0,
        )

    return _cluster_weighted_average(
        features=features,
        labels=labels,
        ignore_noise=True,
    )


def isolation_forest(
    feature_dict: FeatureDict,
    contamination: str | float,
    random_state: int,
) -> np.ndarray:
    features = flatten_features(feature_dict)

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=random_state,
    )

    labels = model.fit_predict(features)

    normal_features = features[labels == 1]

    if len(normal_features) == 0:
        print("[Warning] Isolation Forest removed " "all samples. Falling back to average.")

        return np.mean(
            features,
            axis=0,
        )

    return np.mean(
        normal_features,
        axis=0,
    )


def _cluster_weighted_average(
    features: np.ndarray,
    labels: np.ndarray,
    ignore_noise: bool,
) -> np.ndarray:
    valid_mask = np.ones(
        len(labels),
        dtype=bool,
    )

    if ignore_noise:
        valid_mask = labels != -1

    valid_features = features[valid_mask]
    valid_labels = labels[valid_mask]

    if len(valid_features) == 0:
        raise ValueError("No valid feature vectors were found.")

    label_counts = Counter(valid_labels)

    weights = np.array(
        [label_counts[label] for label in valid_labels],
        dtype=np.float64,
    )

    return np.average(
        valid_features,
        axis=0,
        weights=weights,
    )
