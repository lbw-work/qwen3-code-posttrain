"""在无网络的 Docker 容器里执行模型原始输出，分别保存两套题的结果。"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from generate_eval import ROOT, sha256


def docker_command(run_dir: Path, dataset_dir: Path) -> list[str]:
    """限制网络、文件系统和进程数；模型输出是未知代码，不能直接在宿主机运行。"""
    return [
        "docker", "run", "--rm", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "256", "--memory", "8g", "--cpus", "4",
        "--tmpfs", "/tmp:rw,exec,size=1g",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "-e", "HOME=/tmp",
        "-e", "HUMANEVAL_OVERRIDE_PATH=/datasets/HumanEvalPlus-v0.1.10.jsonl",
        "-e", "MBPP_OVERRIDE_PATH=/datasets/MbppPlus-v0.2.0.jsonl",
        "-v", f"{run_dir}:/work:rw",
        "-v", f"{dataset_dir}:/datasets:ro",
        "qwen-code-eval:0.3.1",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="generate_eval.py 生成的结果目录")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir() or not run_dir.is_relative_to((ROOT / "results").resolve()):
        parser.error("请传入本项目 results/ 下的运行目录")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config["limit_per_suite"] is not None:
        parser.error("冒烟测试只验证生成，不计入完整榜单")

    lock = json.loads((ROOT / "eval.lock.json").read_text(encoding="utf-8"))
    if sha256(ROOT / "eval.jsonl") != lock["eval_jsonl_sha256"]:
        raise ValueError("评测清单校验失败")
    dataset_dir = ROOT / "data" / "evalplus"
    for dataset in lock["datasets"].values():
        if sha256(ROOT / dataset["file"]) != dataset["sha256"]:
            raise ValueError(f"题库校验失败：{dataset['file']}")

    for suite in ("humaneval", "mbpp"):
        samples = run_dir / f"{suite}.samples.jsonl"
        if not samples.is_file():
            raise FileNotFoundError(samples)
        result = run_dir / f"{suite}.samples_eval_results.json"
        if result.exists():
            raise FileExistsError(f"已有评测结果，不覆盖：{result}")

        # 直接判原始代码：EvalPlus 的 sanitize 会删除部分合法辅助函数，
        # 已确认它会让官方 MBPP 标准答案被误判。所有模型都使用相同的原始输出口径。
        prefix = docker_command(run_dir, dataset_dir)
        started = time.monotonic()
        completed = subprocess.run(
            prefix + ["python", "-m", "evalplus.evaluate", "--dataset", suite,
                      "--samples", f"/work/{samples.name}", "--parallel", "2"],
            check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        (run_dir / f"{suite}.judge.log").write_text(completed.stdout, encoding="utf-8")
        (run_dir / f"{suite}.judge.timing.json").write_text(
            json.dumps({"elapsed_seconds": time.monotonic() - started}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(completed.stdout)
    print("两套题均已判分；原始逐题结果和日志保留在", run_dir)


if __name__ == "__main__":
    main()
