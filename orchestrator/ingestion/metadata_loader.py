"""Helpers to load document metadata manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_metadata_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        raise ValueError(f"Unsupported metadata file format: {path}")

    if not isinstance(data, dict):
        raise ValueError("Metadata manifest must be a mapping")

    return data

