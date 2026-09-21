"""从各阶段已冻结的 JSONL 数据集中抽取固定随机样例，供人工阅读格式。"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from generate_eval import ROOT, sha256


# 这些是“训练阶段 -> 实际读取的数据文件”的关系。Full SFT 和 LoRA SFT 的
# 数据完全相同，DPO 和 Reward Model 的数据也完全相同；清单保留这种复用关系，
# 使读者不会误以为每个阶段都另外准备了一份题目。
DATASETS = {
    "evaluation": ROOT / "eval.jsonl",
    "sft": ROOT / "data" / "sft" / "train.jsonl",
    "preference": ROOT / "data" / "preference" / "train.jsonl",
    "rl": ROOT / "data" / "rl" / "train.jsonl",
}
STAGES = {
    "base": "evaluation",
    "full_sft": "sft",
    "lora_sft": "sft",
    "dpo": "preference",
    "reward_model": "preference",
    "ppo": "rl",
    "grpo": "rl",
}


def sample_rows(path: Path, count: int, seed: int) -> list[dict]:
    """从一个 JSONL 文件无放回地抽取 ``count`` 条确定性随机样本。

    JSONL 的每一行本身就是一个样本，因此先逐行解析成字典，再让独立的
    ``random.Random(seed)`` 抽取下标。独立随机对象意味着不会受程序其他位置的
    随机调用影响：同一数据文件、样本数和种子总会得到同一批样例，方便两台机器
    对照。样例文件保留完整字段，不截断代码或测试，读者看到的结构与训练器读取的
    结构完全一致。
    """
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) < count:
        raise ValueError(f"{path} 只有 {len(rows)} 条，无法抽取 {count} 条不重复样本")
    return random.Random(seed).sample(rows, count)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """将完整样本写成 UTF-8 JSONL；一行对应一个可独立阅读或解析的样本。"""
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """生成 ``data/examples`` 下的样例与阶段映射清单。

    只有真正存在的已冻结数据才会被抽样。若偏好数据尚未执行
    ``export_themis.py -> prepare_preference.py``，清单会明确标记它不可用，而不会
    用手写或合成记录凑数。这样读者既能看见已准备数据的真实格式，也能准确知道
    哪条训练路线还缺少数据准备。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10, help="每种实际数据格式抽取的行数")
    parser.add_argument("--seed", type=int, default=20260921, help="固定随机种子")
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count 必须为正整数")

    output = ROOT / "data" / "examples"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"seed": args.seed, "count_per_dataset": args.count, "datasets": {}, "stages": STAGES}
    for name, source in DATASETS.items():
        if not source.is_file():
            manifest["datasets"][name] = {
                "available": False,
                "source": str(source.relative_to(ROOT)),
                "reason": "尚未冻结；请先完成对应的数据准备脚本",
            }
            continue
        # 在种子中加入数据集名称，避免不同格式因为碰巧行数相同而抽中同一组序号。
        dataset_seed = args.seed ^ int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        rows = sample_rows(source, args.count, dataset_seed)
        target = output / f"{name}.sample.jsonl"
        write_jsonl(target, rows)
        manifest["datasets"][name] = {
            "available": True,
            "source": str(source.relative_to(ROOT)),
            "source_sha256": sha256(source),
            "sample": str(target.relative_to(ROOT)),
            "sample_sha256": sha256(target),
            "schema": sorted(rows[0]),
        }

    lock_path = output / "samples.lock.json"
    lock_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
