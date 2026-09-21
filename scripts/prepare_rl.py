"""从已冻结的 SFT 源数据取测试，只给 PPO/GRPO 提供题干和训练测试。"""

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as parquet

from generate_eval import ROOT, sha256


def load_sft_rows(path: Path) -> dict[str, dict]:
    """读取一个 SFT JSONL，并按上游 ``source_id`` 建立随机访问索引。

    RL 数据不重新挑选题目，而是从已经通过 SFT 过滤的样本中取题干和单元测试。
    返回 ``{source_id: sft_row}``，主函数可据此保留原来的训练/验证归属。
    """
    with path.open(encoding="utf-8") as stream:
        return {row["source_id"]: row for row in map(json.loads, stream)}


def main() -> None:
    """从 SFT 同一原始 Parquet 提取 PPO/GRPO 可用的训练测试。

    先用 SHA-256 确认参数给出的 Parquet 正是 SFT 用过的那一份，再读 SFT 的 train/valid
    行，保证不会把验证题移动回训练。每个测试必须是字符串、可被 Python AST 解析、数量
    在 1--20 之间。输出只含 ``prompt`` 与 ``tests``；它不含 EvalPlus 测试，也不含标准
    答案，故 GRPO 的执行奖励无法泄漏正式评测答案。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, required=True, help="prepare_sft.py 使用的同一 Parquet 文件")
    args = parser.parse_args()
    source = args.source_file.resolve()
    sft_lock = json.loads((ROOT / "data" / "sft.lock.json").read_text(encoding="utf-8"))
    if sha256(source) != sft_lock["source_sha256"]:
        raise ValueError("RL 原始数据不是 SFT 使用的固定源文件")
    train_rows = load_sft_rows(ROOT / "data" / "sft" / "train.jsonl")
    valid_rows = load_sft_rows(ROOT / "data" / "sft" / "valid.jsonl")
    if set(train_rows) & set(valid_rows):
        raise ValueError("SFT 训练/验证 source_id 重叠")
    output = ROOT / "data" / "rl"
    lock_path = ROOT / "data" / "rl.lock.json"
    if output.exists() or lock_path.exists():
        raise FileExistsError("RL 数据已准备，不自动覆盖")
    output.mkdir(parents=True)
    counts = Counter()
    with (output / "train.jsonl").open("w", encoding="utf-8") as train, (output / "valid.jsonl").open("w", encoding="utf-8") as valid:
        for batch in parquet.ParquetFile(source).iter_batches(batch_size=1024):
            for raw in batch.to_pylist():
                source_id = raw["id"]
                sft = train_rows.get(source_id) or valid_rows.get(source_id)
                if sft is None:
                    continue
                tests = json.loads(raw["unit_tests"])
                if not isinstance(tests, list) or not tests or len(tests) > 20:
                    counts["invalid_tests"] += 1
                    continue
                try:
                    for test in tests:
                        if not isinstance(test, str):
                            raise ValueError("测试必须是 Python 字符串")
                        ast.parse(test)
                except (SyntaxError, ValueError):
                    counts["invalid_tests"] += 1
                    continue
                # 这里只把训练数据自己带的测试交给 RL；EvalPlus 官方隐藏测试
                # 始终留在独立评测侧，不能用来生成 GRPO 的 reward。
                stream = train if source_id in train_rows else valid
                row = {"source_id": source_id, "prompt": sft["prompt"], "tests": tests}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["train" if stream is train else "valid"] += 1
    lock = {
        "source_sha256": sft_lock["source_sha256"],
        "sft_lock_sha256": sha256(ROOT / "data" / "sft.lock.json"),
        "eval_sha256": sft_lock["eval_sha256"],
        "counts": dict(counts),
        "train_sha256": sha256(output / "train.jsonl"),
        "valid_sha256": sha256(output / "valid.jsonl"),
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
