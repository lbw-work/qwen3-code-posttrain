"""同一份训练/验证数据分别运行 Full SFT 和 LoRA SFT。正式训练需单张 4090。"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from generate_eval import ROOT, sha256


def checked_data() -> tuple[Path, Path, dict]:
    """验证 SFT 的训练、验证数据及其关联评测版本。

    返回 ``(train_jsonl, valid_jsonl, lock)``。训练文件本身与锁内 SHA-256 都要一致，
    同时当前 eval.jsonl 也必须等于准备数据时的版本；否则即便 loss 正常，实验之间也
    无法证明使用了同一组训练数据和同一套污染检查目标。
    """
    folder = ROOT / "data" / "sft"
    lock = json.loads((ROOT / "data" / "sft.lock.json").read_text(encoding="utf-8"))
    train, valid = folder / "train.jsonl", folder / "valid.jsonl"
    for path, expected in ((train, lock["train_sha256"]), (valid, lock["valid_sha256"])):
        if sha256(path) != expected:
            raise ValueError(f"训练数据校验失败：{path}")
    if sha256(ROOT / "eval.jsonl") != lock["eval_sha256"]:
        raise ValueError("评测题库与准备 SFT 数据时不同")
    return train, valid, lock


def main() -> None:
    """运行一次 Full SFT 或 LoRA SFT，并保存可供后续评测的最终模型。

    ``--mode full`` 更新所有 Base 参数，``--mode lora`` 只训练插入线性层的低秩矩阵。
    两者用同一份 prompt/completion 数据、相同的序列长度与验证规则。SFTTrainer 将一条
    样本拼成 ``[prompt tokens, completion tokens, EOS]``，并令 prompt 的 labels 为 -100，
    所以交叉熵只惩罚代码答案。训练结束后保存 final_model 及 training_meta.json；后者
    是后续评测和最终横向比较追溯来源的依据。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("full", "lora"), required=True)
    parser.add_argument("--run-id", help="独立运行编号；默认 UTC 时间")
    parser.add_argument("--max-steps", type=int, default=-1, help="默认完整一轮；正数用于小步试跑")
    args = parser.parse_args()
    if args.max_steps == 0 or args.max_steps < -1:
        parser.error("--max-steps 必须是 -1 或正整数")
    if torch.cuda.device_count() != 1:
        parser.error("正式训练要求一张 CUDA 显卡；本机只检查脚本与小样本")
    if not torch.cuda.is_bf16_supported():
        parser.error("当前显卡不支持 bf16")

    train_file, valid_file, data_lock = checked_data()
    base = ROOT / "models" / "base"
    stage = "full_sft" if args.mode == "full" else "lora_sft"
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ROOT / "runs" / stage / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    dataset = load_dataset("json", data_files={"train": str(train_file), "valid": str(valid_file)})
    # SFTTrainer 只需要 prompt/completion 两列。它会拼接 token 序列，
    # 在 prompt 对应位置把 labels 设为 -100；交叉熵忽略 -100，
    # 因此模型学习的是“题目后面写代码”，而不是背诵题目本身。
    train = dataset["train"].select_columns(["prompt", "completion"])
    valid = dataset["valid"].select_columns(["prompt", "completion"])
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
    )
    model.config.use_cache = False  # 梯度检查点与 KV cache 不应同时开启。

    lora = None
    if args.mode == "lora":
        # all-linear 覆盖注意力和 MLP 的线性层，PEFT 自动跳过输出头。
        # 底座参数冻结，仅训练附加的低秩矩阵；后续 DPO/PPO/GRPO 都从此适配器分叉。
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM")

    settings = SFTConfig(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=2e-5 if args.mode == "full" else 1e-4,
        lr_scheduler_type="cosine",
        # 当前固定的 Transformers 版本用小于 1 的 warmup_steps 表示训练总步数的比例。
        # 0.03 因此仍是前 3% 的优化步用于学习率预热；整数才表示固定步数。
        warmup_steps=0.03,
        optim="paged_adamw_8bit" if args.mode == "full" else "adamw_torch",
        max_length=1024,
        completion_only_loss=True,
        packing=False,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        report_to="none",
        seed=42,
    )
    # 正式模型选择只看 SFT 验证集 loss，不看 HumanEval+/MBPP+。
    # 公开评测集在每个阶段训练结束后只运行一次，避免测试集反复调参。
    trainer = SFTTrainer(
        model=model, args=settings, train_dataset=train, eval_dataset=valid,
        processing_class=tokenizer, peft_config=lora,
    )
    outcome = trainer.train()
    final = run_dir / "final_model"
    trainer.save_model(final)
    tokenizer.save_pretrained(final)
    metadata = {
        "stage": stage,
        "base_model": json.loads((ROOT / "models.lock.json").read_text(encoding="utf-8"))["base"],
        "sft_lock_sha256": sha256(ROOT / "data" / "sft.lock.json"),
        "train_sha256": data_lock["train_sha256"],
        "valid_sha256": data_lock["valid_sha256"],
        "eval_sha256": data_lock["eval_sha256"],
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "training_loss": outcome.training_loss,
        "train_runtime_seconds": outcome.metrics.get("train_runtime"),
        "global_step": trainer.state.global_step,
        "settings": settings.to_dict(),
    }
    (run_dir / "training_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{stage} 完成，推理模型目录：{final}")


if __name__ == "__main__":
    main()
