"""Content extractors for engineering documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader


@dataclass
class PDFTextExtractor:
    """Simple PDF extractor that returns raw text per page."""

    include_empty: bool = False

    def extract_text(self, path: Path) -> List[str]:
        reader = PdfReader(str(path))
        pages: List[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text or self.include_empty:
                pages.append(text)
        return pages


def extract_pdf_text(path: Path, *, include_empty: bool = False) -> List[str]:
    """Convenience wrapper for quick extraction."""

    return PDFTextExtractor(include_empty=include_empty).extract_text(path)


def extract_pdf_tables(path: Path) -> List[str]:
    """Placeholder for table extraction.

    In future iterations we can integrate camelot/tabula to return CSV/Markdown tables.
    For now we simply return an empty list to keep the API stable.
    """

    return []

