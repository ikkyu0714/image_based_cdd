from __future__ import annotations

import re
import shutil

from pathlib import Path

from imagedl.modules.sources import BingImageClient


class ImageDownloader:

    def __init__(
        self,
        download_dir: str | Path,
        images_per_query: int,
        use_hypernyms: bool,
    ):
        self.download_dir = Path(download_dir)
        self.images_per_query = images_per_query
        self.use_hypernyms = use_hypernyms

        self.download_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _flatten_download_directory(
        self,
        output_dir: Path,
    ) -> None:
        bing_dir = output_dir / "BingImageClient"

        if not bing_dir.exists():
            return

        valid_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }

        for image_path in bing_dir.rglob("*"):
            if not image_path.is_file():
                continue

            if image_path.suffix.lower() not in valid_extensions:
                continue

            destination = output_dir / image_path.name

            counter = 1
            while destination.exists():
                destination = output_dir / f"{image_path.stem}_{counter}{image_path.suffix}"
                counter += 1

            image_path.rename(destination)

        shutil.rmtree(bing_dir)

    def download_language_images(
        self,
        language_name: str,
        keywords: list[str],
        hypernyms: list[str],
    ) -> Path:
        language_dir = self.download_dir / language_name

        language_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        queries = self._build_queries(
            keywords=keywords,
            hypernyms=hypernyms,
        )

        for query in queries:
            query_dir = language_dir / self._sanitize_directory_name(query)

            query_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            existing_count = self._count_images(query_dir)

            if existing_count >= self.images_per_query:
                print(f"[Skip] {query}: " f"{existing_count} images already exist")
                continue

            remaining_count = self.images_per_query - existing_count

            print(f"[Download] {query}: " f"{remaining_count} images")

            self._download(
                query=query,
                output_dir=query_dir,
                limit=remaining_count,
            )

        return language_dir

    def _build_queries(
        self,
        keywords: list[str],
        hypernyms: list[str],
    ) -> list[str]:
        if not keywords:
            return []

        if not self.use_hypernyms or not hypernyms:
            return keywords

        queries: list[str] = []

        for keyword in keywords:
            for hypernym in hypernyms:
                query = f"{keyword} {hypernym}"
                queries.append(query)

        return queries

    def _download(
        self,
        query: str,
        output_dir: Path,
        limit: int,
    ) -> None:
        client = BingImageClient(
            work_dir=str(output_dir),
        )

        try:
            image_infos = client.search(
                query,
                search_limits=limit,
                num_threadings=1,
            )

            client.download(
                image_infos,
                num_threadings=1,
            )

            self._flatten_download_directory(output_dir)

        except Exception as error:
            print(f"[Warning] Failed to download " f"images for '{query}': {error}")

    @staticmethod
    def _count_images(directory: Path) -> int:
        valid_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        }

        return sum(1 for path in directory.iterdir() if path.is_file() and path.suffix.lower() in valid_extensions)

    @staticmethod
    def _sanitize_directory_name(name: str) -> str:
        sanitized = re.sub(
            r'[\\/:*?"<>|]',
            "_",
            name,
        )

        return sanitized.strip()
