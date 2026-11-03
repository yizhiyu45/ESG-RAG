import json

from esg_rag.preflight import check_dataset


def test_dataset_preflight_accepts_minimal_inputs(tmp_path):
    (tmp_path / "settings.yaml").write_text("chunks: {}\n", encoding="utf-8")
    qa_dir = tmp_path / "qa-pairs"
    qa_dir.mkdir()
    (qa_dir / "qa-pairs.json").write_text(
        json.dumps([{"id": "q1", "question": "Who?"}]), encoding="utf-8"
    )

    assert check_dataset(tmp_path) == []

