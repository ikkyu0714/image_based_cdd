from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    synset_file: str
    download_dir: str
    start_id: int
    end_id: int


@dataclass
class SearchConfig:
    use_hypernyms: bool
    images_per_query: int


@dataclass
class FeatureConfig:
    model: str
    image_size: tuple[int, int]
    grayscale: bool


@dataclass
class AnalysisConfig:
    methods: list[str]

    kmeans_clusters: int

    dbscan_eps: float
    dbscan_min_samples: int

    isolation_contamination: str | float
    random_state: int


@dataclass
class OutputConfig:
    result_file: str


@dataclass
class Config:
    data: DataConfig
    search: SearchConfig
    feature: FeatureConfig
    analysis: AnalysisConfig
    output: OutputConfig


def load_config(config_path: str | Path) -> Config:
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = yaml.safe_load(file)

    data_config = DataConfig(
        synset_file=raw_config["data"]["synset_file"],
        download_dir=raw_config["data"]["download_dir"],
        start_id=raw_config["data"]["start_id"],
        end_id=raw_config["data"]["end_id"],
    )

    search_config = SearchConfig(
        use_hypernyms=raw_config["search"]["use_hypernyms"],
        images_per_query=raw_config["search"]["images_per_query"],
    )

    image_size = raw_config["feature"]["image_size"]

    feature_config = FeatureConfig(
        model=raw_config["feature"]["model"],
        image_size=(image_size[0], image_size[1]),
        grayscale=raw_config["feature"]["grayscale"],
    )

    analysis_config = AnalysisConfig(
        methods=raw_config["analysis"]["methods"],
        kmeans_clusters=raw_config["analysis"]["kmeans_clusters"],
        dbscan_eps=raw_config["analysis"]["dbscan_eps"],
        dbscan_min_samples=raw_config["analysis"]["dbscan_min_samples"],
        isolation_contamination=raw_config["analysis"]["isolation_contamination"],
        random_state=raw_config["analysis"]["random_state"],
    )

    output_config = OutputConfig(
        result_file=raw_config["output"]["result_file"],
    )

    return Config(
        data=data_config,
        search=search_config,
        feature=feature_config,
        analysis=analysis_config,
        output=output_config,
    )
