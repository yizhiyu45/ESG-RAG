"""Offline environment and dataset checks for ESG-RAG."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


REQUIRED_MODULES = (
    "datashaper",
    "faiss",
    "lancedb",
    "networkx",
    "nltk",
    "numpy",
    "openai",
    "pandas",
    "pyarrow",
    "sklearn",
    "tiktoken",
    "tqdm",
)

INDEX_FILES = (
    "create_final_community_reports.parquet",
    "create_final_entities.parquet",
    "create_final_nodes.parquet",
    "create_final_relationships.parquet",
    "create_final_text_units.parquet",
    "embeddings.entity.description.parquet",
    "embeddings.text_unit.text.parquet",
    "keyword_index.parquet",
    "split_text_units.parquet",
)


def check_environment() -> list[str]:
    """Return actionable environment errors without importing heavy packages."""
    errors: list[str] = []
    if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
        errors.append("Python 3.10-3.12 is required.")
    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        errors.append("Missing modules: " + ", ".join(missing))
    return errors


def check_dataset(root: Path, require_index: bool = False) -> list[str]:
    """Validate the files required before indexing or retrieval."""
    errors: list[str] = []
    if not (root / "settings.yaml").is_file():
        errors.append(f"Missing {root / 'settings.yaml'}")
    qa_path = root / "qa-pairs" / "qa-pairs.json"
    if not qa_path.is_file():
        errors.append(f"Missing {qa_path}")
    else:
        try:
            data = json.loads(qa_path.read_text(encoding="utf-8"))
            if not isinstance(data, list) or not data:
                errors.append(f"{qa_path} must contain a non-empty JSON list.")
            elif any("id" not in item or "question" not in item for item in data):
                errors.append(f"Every item in {qa_path} needs id and question fields.")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"Cannot read {qa_path}: {exc}")
    if require_index:
        missing = [name for name in INDEX_FILES if not (root / "output" / name).is_file()]
        if missing:
            errors.append("Missing index outputs: " + ", ".join(missing))
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ESG-RAG before a run.")
    parser.add_argument("dataset_root", nargs="?", type=Path)
    parser.add_argument(
        "--require-index",
        action="store_true",
        help="also require all index files used by the retrieval stage",
    )
    args = parser.parse_args()
    errors = check_environment()
    if args.dataset_root is not None:
        errors.extend(check_dataset(args.dataset_root, args.require_index))
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        raise SystemExit(1)
    print("[OK] ESG-RAG preflight checks passed.")


if __name__ == "__main__":
    main()

