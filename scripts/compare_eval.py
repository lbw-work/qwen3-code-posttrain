"""读取六个阶段各自的冻结评测，写出可追溯的最终横向对比。"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from generate_eval import ROOT, sha256


REQUIRED = {"base", "full_sft", "lora_sft", "dpo", "ppo", "grpo"}
ORDER = ("base", "reference", "full_sft", "lora_sft", "dpo", "ppo", "grpo")


def read_run(folder: Path) -> dict:
    """读取并严格验证一个正式评测运行，返回写入最终表格的标准化记录。

    检查范围包括：目录属于 results、config 与 summary 的阶段/评测哈希一致、不是冒烟测试、
    生成参数仍为固定贪心 512 token，以及每套逐题结果的 SHA-256 等于摘要记录。只有这些
    前提同时成立，某个阶段的 pass@1 才可与其它阶段放在一张表中。
    """
    folder = folder.resolve()
    if not folder.is_relative_to((ROOT / "results").resolve()):
        raise ValueError(f"不是本项目的结果目录：{folder}")
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    if summary["stage"] != config["stage"] or summary["eval_sha256"] != config["eval_sha256"]:
        raise ValueError(f"配置与摘要版本不同：{folder}")
    if config["limit_per_suite"] is not None:
        raise ValueError(f"冒烟测试不能进入最终比较：{folder}")
    generation = config["generation"]
    if generation != {"do_sample": False, "max_new_tokens": 512}:
        raise ValueError(f"生成参数不一致：{folder}")
    for suite in ("humaneval", "mbpp"):
        result = folder / f"{suite}.samples_eval_results.json"
        if sha256(result) != summary["suites"][suite]["result_sha256"]:
            raise ValueError(f"逐题判分被改动：{result}")
    timing_path = folder / "generation_timing.json"
    timing = json.loads(timing_path.read_text(encoding="utf-8")) if timing_path.exists() else None
    model = Path(config["model"])
    training_meta = model.parent / "training_meta.json"
    training = json.loads(training_meta.read_text(encoding="utf-8")) if training_meta.exists() else None
    return {
        "stage": config["stage"], "run_dir": str(folder), "model": str(model),
        "eval_sha256": summary["eval_sha256"], "generation": generation,
        "human_eval_plus": summary["suites"]["humaneval"],
        "mbpp_plus": summary["suites"]["mbpp"],
        "generation_timing": timing,
        "training_meta": training,
        "config_sha256": sha256(folder / "config.json"),
        "summary_sha256": sha256(folder / "summary.json"),
    }


def main() -> None:
    """汇总 Base、Full SFT、LoRA SFT、DPO、PPO、GRPO 的最终评测。

    调用者显式传入每个阶段的结果目录，函数拒绝重复阶段、缺失阶段或不同 eval.jsonl 版本。
    通过验证后，按固定顺序写 JSON 全量证据和 Markdown pass@1 表。它不计算新分数，只把
    已验证的原始结果组织为最终横向对比，因此不会意外选择“最新但参数不同”的一次运行。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="每阶段一个 results/<stage>/<run-id> 目录")
    parser.add_argument("--comparison-id", help="输出目录名；默认 UTC 时间")
    args = parser.parse_args()
    runs = [read_run(path) for path in args.run_dirs]
    stages = [run["stage"] for run in runs]
    if len(stages) != len(set(stages)):
        parser.error("同一阶段只能传入一次；多次重复实验请分别生成比较报告")
    if not REQUIRED.issubset(stages):
        parser.error(f"最终比较缺少阶段：{sorted(REQUIRED - set(stages))}")
    eval_shas = {run["eval_sha256"] for run in runs}
    if len(eval_shas) != 1 or eval_shas.pop() != sha256(ROOT / "eval.jsonl"):
        raise ValueError("各阶段没有使用同一套冻结评测题")
    runs.sort(key=lambda row: ORDER.index(row["stage"]))
    name = args.comparison_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "results" / "comparisons" / name
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(json.dumps(runs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 各阶段代码通过率对比", "",
        "| 阶段 | HumanEval+ pass@1 | MBPP+ pass@1 | 生成耗时（秒） |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in runs:
        human = row["human_eval_plus"]
        mbpp = row["mbpp_plus"]
        elapsed = row["generation_timing"]["generation_elapsed_seconds"] if row["generation_timing"] else None
        lines.append(
            f"| {row['stage']} | {human['passed']}/{human['total']} ({human['pass_at_1']:.1%}) "
            f"| {mbpp['passed']}/{mbpp['total']} ({mbpp['pass_at_1']:.1%}) "
            f"| {elapsed:.1f} |" if elapsed is not None else
            f"| {row['stage']} | {human['passed']}/{human['total']} ({human['pass_at_1']:.1%}) "
            f"| {mbpp['passed']}/{mbpp['total']} ({mbpp['pass_at_1']:.1%}) | 未记录 |"
        )
    lines += [
        "", "表中直接比较代码通过率。每个阶段的逐题输出、判分、模型与数据版本见 comparison.json 指向的独立结果目录。",
        "DPO、PPO、GRPO 的计算量需结合各自 training_meta.json 解读；PPO 路线还需计入奖励模型训练耗时。",
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
