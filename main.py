from __future__ import annotations

import argparse

from cultural_difference.config import load_config
from cultural_difference.detector import CulturalDifferenceDetector
from cultural_difference.feature_extractor import VGG16FeatureExtractor
from cultural_difference.image_downloader import ImageDownloader
from cultural_difference.result_writer import CsvResultWriter
from cultural_difference.synset_repository import SynsetRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Image-based cultural difference detector")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    synset_repository = SynsetRepository(config.data.synset_file)

    image_downloader = ImageDownloader(
        download_dir=config.data.download_dir,
        images_per_query=config.search.images_per_query,
        use_hypernyms=config.search.use_hypernyms,
    )

    feature_extractor = VGG16FeatureExtractor(
        grayscale=config.feature.grayscale,
    )

    result_writer = CsvResultWriter(config.output.result_file)

    detector = CulturalDifferenceDetector(
        image_downloader=image_downloader,
        feature_extractor=feature_extractor,
        result_writer=result_writer,
        config=config,
    )

    synsets = synset_repository.load()

    for synset_id in range(
        config.data.start_id,
        config.data.end_id + 1,
    ):
        if synset_id not in synsets:
            print(f"Synset ID {synset_id} was not found.")
            continue

        synset = synsets[synset_id]

        print("=" * 60)
        print(f"ID: {synset_id}")
        print(f"Japanese: {synset.japanese_keyword}")
        print(f"English : {synset.english_keyword}")
        print("=" * 60)

        detector.run(
            synset_id=synset_id,
            synset=synset,
        )


if __name__ == "__main__":
    main()
