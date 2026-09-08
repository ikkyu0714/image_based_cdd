from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.utils import load_img, img_to_array


class VGG16FeatureExtractor:
    def __init__(
        self,
        grayscale: bool = False,
        image_size: tuple[int, int] = (224, 224),
    ):
        self.grayscale = grayscale
        self.image_size = image_size

        self.model = VGG16(
            include_top=False,
            weights="imagenet",
        )

    def extract_directory_features(
        self,
        root_dir: str | Path,
    ) -> dict[str, list[np.ndarray]]:
        root_dir = Path(root_dir)

        feature_dict: dict[str, list[np.ndarray]] = {}

        if not root_dir.exists():
            return feature_dict

        for query_dir in sorted(root_dir.iterdir()):
            if not query_dir.is_dir():
                continue

            features = self._extract_query_features(query_dir)

            if features:
                feature_dict[query_dir.name] = features

        return feature_dict

    def _extract_query_features(
        self,
        query_dir: Path,
    ) -> list[np.ndarray]:
        features: list[np.ndarray] = []

        for image_path in sorted(query_dir.iterdir()):
            if not image_path.is_file():
                continue

            if not self._is_image(image_path):
                continue

            try:
                feature = self.extract_image_feature(image_path)

                features.append(feature)

            except Exception as error:
                print(f"[Warning] Failed to extract feature " f"from {image_path}: {error}")

        return features

    def extract_image_feature(
        self,
        image_path: str | Path,
    ) -> np.ndarray:
        image_path = Path(image_path)

        if self.grayscale:
            input_image = self._load_grayscale_image(image_path)
        else:
            input_image = load_img(
                image_path,
                target_size=self.image_size,
            )

            input_image = img_to_array(input_image)

        input_batch = np.expand_dims(
            input_image,
            axis=0,
        )

        input_batch = preprocess_input(input_batch)

        feature_map = self.model.predict(
            input_batch,
            verbose=0,
        )

        feature_vector = feature_map.reshape(-1)

        return feature_vector

    def _load_grayscale_image(
        self,
        image_path: Path,
    ) -> np.ndarray:
        gray_image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if gray_image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        gray_image = cv2.resize(
            gray_image,
            self.image_size,
        )

        gray_image = self._gamma_correction(gray_image)

        rgb_image = cv2.cvtColor(
            gray_image,
            cv2.COLOR_GRAY2RGB,
        )

        return rgb_image.astype(np.float32)

    @staticmethod
    def _gamma_correction(
        gray_image: np.ndarray,
    ) -> np.ndarray:
        mean = np.mean(gray_image)

        if mean <= 0:
            return gray_image

        gamma = np.log10(0.5) / np.log10(mean / 255.0)

        corrected = 255.0 * (gray_image / 255.0) ** gamma

        return np.clip(
            corrected,
            0,
            255,
        ).astype(np.uint8)

    @staticmethod
    def _is_image(path: Path) -> bool:
        valid_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }

        return path.suffix.lower() in valid_extensions
