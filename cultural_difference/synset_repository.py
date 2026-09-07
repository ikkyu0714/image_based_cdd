from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import openpyxl


@dataclass
class SynsetEntry:
    japanese_hypernyms: list[str]
    japanese_keywords: list[str]
    english_hypernyms: list[str]
    english_keywords: list[str]

    @property
    def japanese_keyword(self) -> str:
        return ", ".join(self.japanese_keywords)

    @property
    def english_keyword(self) -> str:
        return ", ".join(self.english_keywords)


class SynsetRepository:
    def __init__(self, synset_file: str | Path):
        self.synset_file = Path(synset_file)

    def load(self) -> dict[int, SynsetEntry]:
        workbook = openpyxl.load_workbook(
            self.synset_file,
            read_only=True,
            data_only=True,
        )

        worksheet = workbook.worksheets[0]

        synsets: dict[int, SynsetEntry] = {}

        for synset_id, row in enumerate(
            worksheet.iter_rows(
                min_row=2,
                min_col=2,
                values_only=True,
            ),
            start=1,
        ):
            japanese_hypernyms = self._parse_cell(row[0])
            japanese_keywords = self._parse_cell(row[1])
            english_hypernyms = self._parse_cell(row[2])
            english_keywords = self._parse_cell(row[3])

            synsets[synset_id] = SynsetEntry(
                japanese_hypernyms=japanese_hypernyms,
                japanese_keywords=japanese_keywords,
                english_hypernyms=english_hypernyms,
                english_keywords=english_keywords,
            )

        workbook.close()

        return synsets

    @staticmethod
    def _parse_cell(value: object) -> list[str]:
        if value is None:
            return []

        text = str(value).replace("_", " ")

        return [item.strip() for item in text.split(",") if item.strip()]
