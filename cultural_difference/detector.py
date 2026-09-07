from __future__ import annotations

from pathlib import Path

import numpy as np

from cultural_difference import aggregators
from cultural_difference.config import Config
from cultural_difference.feature_extractor import VGG16FeatureExtractor
from cultural_difference.image_downloader import GoogleImageDownloader
from cultural_difference.metrics import (
    cosine_distance,
    cosine_similarity,
    mean_pairwise_euclidean_distance,
)
from cultural_difference.result_writer import CsvResultWriter
from cultural_difference.synset_repository import SynsetEntry


class CulturalDifferenceDetector:
    def __init__(
        self,
        image_downloader: GoogleImageDownloader,
        feature_extractor: VGG16FeatureExtractor,
        result_writer: CsvResultWriter,
        config: Config,
    ):
        self.image_downloader = image_downloader
        self.feature_extractor = feature_extractor
        self.result_writer = result_writer
        self.config = config

    def run(
        self,
        synset_id: int,
        synset: SynsetEntry,
    ) -> None:
        japanese_dir = self.image_downloader.download_language_images(
            language_name=f"{synset_id}/japanese",
            keywords=synset.japanese_keywords,
            hypernyms=synset.japanese_hypernyms,
        )

        english_dir = self.image_downloader.download_language_images(
            language_name=f"{synset_id}/english",
            keywords=synset.english_keywords,
            hypernyms=synset.english_hypernyms,
        )

        japanese_feature_dict = self.feature_extractor.extract_directory_features(japanese_dir)

        english_feature_dict = self.feature_extractor.extract_directory_features(english_dir)

        if not japanese_feature_dict:
            print(f"[Warning] No Japanese features " f"for Synset ID {synset_id}")
            return

        if not english_feature_dict:
            print(f"[Warning] No English features " f"for Synset ID {synset_id}")
            return

        japanese_features = aggregators.flatten_features(japanese_feature_dict)

        english_features = aggregators.flatten_features(english_feature_dict)

        japanese_mean_euclidean = mean_pairwise_euclidean_distance(japanese_features)

        english_mean_euclidean = mean_pairwise_euclidean_distance(english_features)

        print(f"Japanese images: " f"{len(japanese_features)}")

        print(f"English images: " f"{len(english_features)}")

        for method in self.config.analysis.methods:
            try:
                japanese_vector = self._aggregate(
                    feature_dict=japanese_feature_dict,
                    method=method,
                )

                english_vector = self._aggregate(
                    feature_dict=english_feature_dict,
                    method=method,
                )

            except ValueError as error:
                print(f"[Warning] {method}: {error}")
                continue

            similarity = cosine_similarity(
                japanese_vector,
                english_vector,
            )

            distance = cosine_distance(
                japanese_vector,
                english_vector,
            )

            print(f"[{method}] " f"cosine similarity = " f"{similarity:.6f}")

            print(f"[{method}] " f"cosine distance   = " f"{distance:.6f}")

            self.result_writer.write(
                synset_id=synset_id,
                japanese_keyword=synset.japanese_keyword,
                english_keyword=synset.english_keyword,
                method=method,
                cosine_similarity=similarity,
                cosine_distance=distance,
                japanese_mean_pairwise_euclidean=(japanese_mean_euclidean),
                english_mean_pairwise_euclidean=(english_mean_euclidean),
                japanese_image_count=len(japanese_features),
                english_image_count=len(english_features),
            )

    def _aggregate(
        self,
        feature_dict: dict[str, list[np.ndarray]],
        method: str,
    ) -> np.ndarray:
        if method == "average":
            return aggregators.average(feature_dict)

        if method == "centroid_of_centroids":
            return aggregators.centroid_of_centroids(feature_dict)

        if method == "kmeans":
            return aggregators.kmeans(
                feature_dict=feature_dict,
                n_clusters=(self.config.analysis.kmeans_clusters),
                random_state=(self.config.analysis.random_state),
            )

        if method == "dbscan":
            return aggregators.dbscan(
                feature_dict=feature_dict,
                eps=self.config.analysis.dbscan_eps,
                min_samples=(self.config.analysis.dbscan_min_samples),
            )

        if method == "isolation_forest":
            return aggregators.isolation_forest(
                feature_dict=feature_dict,
                contamination=(self.config.analysis.isolation_contamination),
                random_state=(self.config.analysis.random_state),
            )

        raise ValueError(f"Unknown aggregation method: {method}")
