from pathlib import Path

from small.corpus import load_jsonl


def test_load_jsonl(tmp_path: Path):
    data = """{"id":"doc1","text":"hello world","domain":"policy"}\n{"id":"doc2","text":"print(1)","domain":"code","language":"python"}\n"""
    path = tmp_path / "sample.jsonl"
    path.write_text(data, encoding="utf-8")

    docs = list(load_jsonl(path))
    assert len(docs) == 2
    assert docs[0].id == "doc1"
    assert docs[1].language == "python"
