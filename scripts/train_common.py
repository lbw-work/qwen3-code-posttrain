"""后训练各阶段共用的路径、数据校验；奖励模型还使用 Trainer 时间回调。"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from transformers import TrainerCallback

from generate_eval import ROOT, sha256


def new_run(stage: str, run_id: str | None) -> Path:
    """创建一次训练专属的空目录，返回其绝对路径。

    未传 ``run_id`` 时使用 UTC 时间戳；目录用 ``exist_ok=False`` 创建，所以同名运行
    会立即失败而非覆盖已有 checkpoint、日志或元数据。所有后训练脚本借此保持相同的
    ``runs/<stage>/<run-id>/`` 布局。
    """
    name = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = ROOT / "runs" / stage / name
    path.mkdir(parents=True, exist_ok=False)
    return path


def sft_adapter(sft_run: Path) -> Path:
    """验证并返回一条 LoRA SFT 运行的 ``final_model`` 适配器目录。

    DPO、PPO、GRPO 必须从同一个 LoRA SFT 分叉，这样差异才能归因于优化方法。此函数
    检查路径属于 ``runs/lora_sft``、元数据 stage 确为 lora_sft、适配器配置文件存在，
    任一条件不满足就停止，避免把 DPO 接在 PPO 等错误起点之后。
    """
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
    """返回已验证的 train/valid JSONL 路径和锁字典。

    ``name`` 决定读取 ``data/<name>.lock.json``。函数比较两个数据文件和当前
    ``eval.jsonl`` 的摘要；这把“训练数据版本”和“当时要避开的评测版本”绑定在一起。
    返回值顺序固定为 ``(train_path, valid_path, lock)``。
    """
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
    """将本阶段的模型来源、数据摘要、超参数和训练结果写成 JSON 元数据。"""
    (run / "training_meta.json").write_text(
        json.dumps(fields, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


class TimeBudget(TrainerCallback):
    """在奖励模型 Trainer 的 step 边界停止训练，限制墙钟预算。

    它不强杀正在进行的一个 step：on_step_end 才设置 ``should_training_stop``，以便
    Trainer 仍能正常保存状态。``time.monotonic`` 不受系统时钟调整影响；实际运行时长
    仍由训练元数据记录，不能把墙钟上限误解为精确 GPU 计算时长。
    """

    def __init__(self, max_hours: float):
        """把用户给出的小时数一次换算为秒，并延后到训练开始时再记起点。"""
        self.seconds = max_hours * 3600
        self.started = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Trainer 真正开始训练时记录单调时钟，避免模型加载时间占用预算。"""
        self.started = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        """当已用时间达到预算时请求 Trainer 在当前 step 后收尾。"""
        if time.monotonic() - self.started >= self.seconds:
            control.should_training_stop = True
        return control
