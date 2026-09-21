"""在无网络的 Docker 容器里执行模型原始输出，分别保存两套题的结果。"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from generate_eval import ROOT, sha256


def docker_command(run_dir: Path, dataset_dir: Path) -> list[str]:
    """构造一次 EvalPlus Docker 调用的公共安全前缀。

    ``run_dir`` 以读写方式挂载，因为 EvalPlus 要写逐题结果；题库只读挂载，防止
    候选代码修改测试。容器没有网络、根文件系统只读、能力被全部删除，并限制 CPU、
    内存与进程数。返回的是参数列表而不是 shell 字符串，因此题目路径中的空格或
    特殊字符不会被 shell 再次解释。
    """
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
    """对一次完整生成结果执行 HumanEval+ 和 MBPP+。

    先验证运行目录、冻结清单和题库哈希，再逐套题调用官方 EvalPlus。输入必须是
    未使用 ``--limit`` 的完整生成结果；否则样本数太小，不能记入正式榜单。这里直接
    传模型原始代码给 EvalPlus：曾验证过其 sanitize 工具会删除合法辅助函数，连官方
    标准答案都可能被误判。每套题的原始逐题 JSON、日志和耗时都留在同一运行目录。
    """
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
