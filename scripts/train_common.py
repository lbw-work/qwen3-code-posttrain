"""DPO、奖励模型、PPO、GRPO 共用的少量路径与数据校验。"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from transformers import TrainerCallback

from generate_eval import ROOT, sha256


def new_run(stage: str, run_id: str | None) -> Path:
    name = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / "runs" / stage / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def sft_adapter(sft_run: Path) -> Path:
    """RL 三条路线从同一个 LoRA SFT 结果分叉，不从彼此的结果继续训练。"""
    sft_run = sft_run.resolve()
    if not sft_run.is_relative_to((ROOT / "runs" / "lora_sft").resolve()):
        raise ValueError("--sft-run 必须是 runs/lora_sft/ 下的一次训练")
    meta = json.loads((sft_run / "training_meta.json").read_text(encoding="utf-8"))
    if meta["stage"] != "lora_sft":
        raise ValueError("所选训练不是 LoRA SFT")
    adapter = sft_run / "final_model"
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"未找到 SFT 适配器：{adapter}")
    return adapter


def locked_jsonl(folder: Path, name: str) -> tuple[Path, Path, dict]:
    """要求有训练、验证数据和哈希锁，避免不同 RL 阶段暗中换数据。"""
    lock_path = ROOT / "data" / f"{name}.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    train, valid = folder / "train.jsonl", folder / "valid.jsonl"
    for file, expected in ((train, lock["train_sha256"]), (valid, lock["valid_sha256"])):
        if sha256(file) != expected:
            raise ValueError(f"数据哈希不匹配：{file}")
    if lock["eval_sha256"] != sha256(ROOT / "eval.jsonl"):
        raise ValueError("数据使用的评测版本与当前项目不一致")
    return train, valid, lock


def save_meta(run: Path, **fields) -> None:
    (run / "training_meta.json").write_text(
        json.dumps(fields, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


class TimeBudget(TrainerCallback):
    """DPO/GRPO 每次训练用同一墙钟上限；日志另外记录实际 GPU 时长。"""

    def __init__(self, max_hours: float):
        self.seconds = max_hours * 3600
        self.started = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() - self.started >= self.seconds:
            control.should_training_stop = True
        return control
