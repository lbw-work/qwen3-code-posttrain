"""把 GRPO 的候选代码送入隔离容器，返回每份代码通过的测试比例。"""

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def score_batch(completions: list[str], tests: list[list[str]]) -> list[float]:
    if len(completions) != len(tests):
        raise ValueError("候选代码与测试必须一一对应")
    samples = [{"code": code, "tests": cases} for code, cases in zip(completions, tests, strict=True)]
    command = [
        "docker", "run", "-i", "--rm", "--network", "none", "--read-only",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--pids-limit", "64", "--memory", "1g", "--cpus", "1",
        "--tmpfs", "/tmp:rw,exec,size=256m", "--user", f"{os.getuid()}:{os.getgid()}",
        "-v", f"{ROOT / 'scripts'}:/runner:ro", "qwen-code-eval:0.3.1",
        "python", "-I", "/runner/code_reward_worker.py",
    ]
    # 只传测试与生成代码，不把模型权重和训练目录挂载进容器。
    try:
        completed = subprocess.run(
            command, input=json.dumps(samples, ensure_ascii=False), text=True,
            capture_output=True, timeout=15 * len(samples) + 30, check=True,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"执行奖励容器失败：{error.stderr.strip()}") from error
    scores = json.loads(completed.stdout)
    if len(scores) != len(samples):
        raise ValueError("容器返回的分数数量不正确")
    return [float(value) for value in scores]
