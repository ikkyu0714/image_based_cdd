from __future__ import annotations

import re
from pathlib import Path

from google_images_download.google_images_download import googleimagesdownload


class GoogleImageDownloader:
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
        downloader = googleimagesdownload()

        arguments = {
            "keywords": query,
            "limit": limit,
            "format": "jpg",
            "output_directory": str(output_dir.parent),
            "image_directory": output_dir.name,
            "silent_mode": True,
        }

        try:
            downloader.download(arguments)

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
