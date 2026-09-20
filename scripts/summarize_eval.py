"""从 EvalPlus 原始逐题结果计算 pass@1，并保留可追溯的摘要。"""

import argparse
import json
from pathlib import Path

from generate_eval import ROOT, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_relative_to((ROOT / "results").resolve()):
        parser.error("请传入本项目 results/ 下的运行目录")
    output = run_dir / "summary.json"
    if output.exists():
        raise FileExistsError(f"已有汇总，不覆盖：{output}")
    lock = json.loads((ROOT / "eval.lock.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config["eval_sha256"] != lock["eval_jsonl_sha256"]:
        raise ValueError("此次运行的评测清单版本与当前冻结版本不一致")

    summary = {"stage": config["stage"], "eval_sha256": config["eval_sha256"], "suites": {}}
    with (ROOT / "eval.jsonl").open(encoding="utf-8") as stream:
        expected_ids = {}
        for line in stream:
            task = json.loads(line)
            expected_ids.setdefault(task["suite"], set()).add(task["task_id"])
    for suite in ("humaneval", "mbpp"):
        path = run_dir / f"{suite}.samples_eval_results.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        per_task = result["eval"]
        expected_count = lock["datasets"][suite]["count"]
        if set(per_task) != expected_ids[suite] or len(per_task) != expected_count or any(len(attempts) != 1 for attempts in per_task.values()):
            raise ValueError(f"{suite} 的题数或每题生成次数不正确")
        # EvalPlus+ 的通过条件是原始测试和新增测试都通过。
        # 每题恰好生成一次，因此通过题数 / 总题数就是本实验的 pass@1。
        passed = sum(
            attempts[0]["base_status"] == "pass" and attempts[0]["plus_status"] == "pass"
            for attempts in per_task.values()
        )
        summary["suites"][suite] = {
            "passed": passed,
            "total": expected_count,
            "pass_at_1": passed / expected_count,
            "result_sha256": sha256(path),
        }
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
