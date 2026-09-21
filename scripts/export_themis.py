"""按固定提交号导出 Themis 的 Python 功能正确性偏好对；可稍后在服务器运行。"""

import json
from collections import Counter
from numbers import Integral

from datasets import load_dataset

from generate_eval import ROOT, sha256


REPO = "project-themis/Themis-CodePreference"
REVISION = "7c366b23590cc9ff8d372bb47280fcd474536344"


def label(feature, value) -> str:
    """把 datasets 的枚举值稳定地转成可读字符串。

    Themis 的 ``language`` 与 ``aspect`` 在 Arrow 数据集中可能以整数编码，
    对应的 ``ClassLabel`` 对象提供 ``int2str``。满足这两个条件时先还原类别名；
    其他数据版本若已经给出字符串，则直接转成字符串。这样主循环可统一比较
    ``"Python"``、``"Functional Correctness"``，不会把编码数字写进导出数据。
    """
    return feature.int2str(int(value)) if hasattr(feature, "int2str") and isinstance(value, Integral) else str(value)


def main() -> None:
    """流式导出 Themis 中适合代码偏好训练的 Python 功能正确性样本。

    数据集很大，因此 ``streaming=True`` 让每次只取一行，不在内存中构造完整
    数据集。循环中先还原语言和评价维度，只保留 ``Python`` 且指标是功能正确性
    的偏好对；每条输出包含同一题目的 ``input``、更优答案 ``chosen``、较差答案
    ``rejected`` 和原始 ``idx``。后续 ``prepare_preference.py`` 会再做代码提取与
    质量过滤。

    ``REVISION`` 是不可变提交号，输出完成后锁文件记录源仓库、提交号、各类行数
    和导出文件哈希。已有导出时拒绝覆盖，避免不知不觉把训练数据换成另一批。
    """
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
