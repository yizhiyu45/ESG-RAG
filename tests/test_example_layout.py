import json
from pathlib import Path


def test_bundled_musique_example_has_expected_layout():
    root = Path(__file__).resolve().parents[1] / "examples" / "musique"
    document = root / "input" / "document.txt"
    qa_file = root / "qa-pairs" / "qa-pairs.json"

    assert document.stat().st_size > 1_000_000
    questions = json.loads(qa_file.read_text(encoding="utf-8"))
    assert len(questions) == 500
    assert all("id" in item and "question" in item for item in questions)

