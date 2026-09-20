"""按固定提交号导出 Themis 的 Python 功能正确性偏好对；可稍后在服务器运行。"""

import json
from collections import Counter
from numbers import Integral

from datasets import load_dataset

from generate_eval import ROOT, sha256


REPO = "project-themis/Themis-CodePreference"
REVISION = "7c366b23590cc9ff8d372bb47280fcd474536344"


def label(feature, value) -> str:
    """HF ClassLabel 在实际行中是整数；导出时转成清楚的类别名。"""
    return feature.int2str(int(value)) if hasattr(feature, "int2str") and isinstance(value, Integral) else str(value)


def main() -> None:
    output = ROOT / "data" / "raw" / "themis-python-fc.jsonl"
    if output.exists():
        raise FileExistsError(f"已有导出，不覆盖：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(REPO, split="train", streaming=True, revision=REVISION)
    counts = Counter()
    with output.open("w", encoding="utf-8") as stream:
        for row in dataset:
            counts["source_rows"] += 1
            language = label(dataset.features["language"], row["language"])
            aspect = label(dataset.features["aspect"], row["aspect"])
            normalized_aspect = "".join(ch for ch in aspect.casefold() if ch.isalnum())
            if language.casefold() != "python" or normalized_aspect not in ("fc", "functionalcorrectness"):
                continue
            record = {
                "language": "Python", "aspect": "Functional Correctness",
                "input": row["input"], "chosen": row["chosen"], "rejected": row["rejected"],
                "source_id": row["idx"],
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["python_functional_correctness"] += 1
    meta = {
        "repo": REPO, "revision": REVISION, "counts": dict(counts),
        "export_sha256": sha256(output),
    }
    (ROOT / "data" / "themis-export.lock.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
