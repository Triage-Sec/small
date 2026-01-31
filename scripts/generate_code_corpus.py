import json
from pathlib import Path


def main(source_dir: str, out_path: str) -> None:
    path = Path(source_dir)
    files = list(path.rglob("*.py"))[:200]
    items = []
    for idx, file_path in enumerate(files):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        items.append(
            {
                "id": f"py_{idx:03d}",
                "text": text,
                "domain": "code",
                "language": "python",
                "source": str(file_path),
            }
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main(".", "tests/fixtures/corpora/code/python_delta.jsonl")
