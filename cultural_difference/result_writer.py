from __future__ import annotations

import csv
from pathlib import Path


class CsvResultWriter:
    def __init__(
        self,
        result_file: str | Path,
    ):
        self.result_file = Path(result_file)

        self.result_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._initialize_file()

    def _initialize_file(self) -> None:
        if self.result_file.exists():
            return

        with self.result_file.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    "synset_id",
                    "japanese_keyword",
                    "english_keyword",
                    "method",
                    "cosine_similarity",
                    "cosine_distance",
                    "japanese_mean_pairwise_euclidean",
                    "english_mean_pairwise_euclidean",
                    "japanese_image_count",
                    "english_image_count",
                ]
            )

    def write(
        self,
        synset_id: int,
        japanese_keyword: str,
        english_keyword: str,
        method: str,
        cosine_similarity: float,
        cosine_distance: float,
        japanese_mean_pairwise_euclidean: float,
        english_mean_pairwise_euclidean: float,
        japanese_image_count: int,
        english_image_count: int,
    ) -> None:
        with self.result_file.open(
            "a",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(
                [
                    synset_id,
                    japanese_keyword,
                    english_keyword,
                    method,
                    cosine_similarity,
                    cosine_distance,
                    japanese_mean_pairwise_euclidean,
                    english_mean_pairwise_euclidean,
                    japanese_image_count,
                    english_image_count,
                ]
            )
