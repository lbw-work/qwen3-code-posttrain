"""从 LoRA SFT 适配器出发，用同一份代码偏好数据运行 DPO。"""

import argparse

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

from generate_eval import ROOT, sha256
from train_common import TimeBudget, locked_jsonl, new_run, save_meta, sft_adapter


def load_sft_copy(adapter, trainable: bool):
    """两份权重起点相同；policy 可训练，reference 从始至终冻结。"""
    base = AutoModelForCausalLM.from_pretrained(
        ROOT / "models" / "base", dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )
    return PeftModel.from_pretrained(base, adapter, is_trainable=trainable)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-run", required=True, type=str)
    parser.add_argument("--run-id")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-hours", type=float, default=4)
    args = parser.parse_args()
    if args.max_hours <= 0:
        parser.error("--max-hours 必须大于 0")
    if torch.cuda.device_count() != 1:
        parser.error("DPO 正式训练要求一张 CUDA 显卡")
    adapter = sft_adapter(ROOT / args.sft_run)
    train_file, valid_file, lock = locked_jsonl(ROOT / "data" / "preference", "preference")
    data = load_dataset("json", data_files={"train": str(train_file), "valid": str(valid_file)})
    for split in ("train", "valid"):
        data[split] = data[split].select_columns(["prompt", "chosen", "rejected"])

    run = new_run("dpo", args.run_id)
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    policy = load_sft_copy(adapter, trainable=True)
    reference = load_sft_copy(adapter, trainable=False)
    settings = DPOConfig(
        output_dir=str(run / "checkpoints"),
        beta=0.1,
        max_length=1024,
        truncation_mode="keep_start",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=5e-6,
        num_train_epochs=1,
        max_steps=args.max_steps,
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
    # DPO 对同一题的 chosen/rejected 分别求 token 对数概率。
    # 相对于冻结 reference，它提高好答案的相对概率、压低坏答案的相对概率；
    # beta 控制偏离 reference 的力度。测试题不参与这个损失。
    trainer = DPOTrainer(
        model=policy, ref_model=reference, args=settings,
        train_dataset=data["train"], eval_dataset=data["valid"],
        processing_class=tokenizer, callbacks=[TimeBudget(args.max_hours)],
    )
    outcome = trainer.train()
    final = run / "final_model"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    save_meta(
        run, stage="dpo", sft_run=str(adapter.parent),
        sft_adapter_config_sha256=sha256(adapter / "adapter_config.json"),
        preference_lock_sha256=sha256(ROOT / "data" / "preference.lock.json"),
        train_sha256=lock["train_sha256"], valid_sha256=lock["valid_sha256"],
        eval_sha256=lock["eval_sha256"],
        best_checkpoint=trainer.state.best_model_checkpoint,
        best_eval_loss=trainer.state.best_metric,
        training_loss=outcome.training_loss, global_step=trainer.state.global_step,
        train_runtime_seconds=outcome.metrics.get("train_runtime"),
        max_hours=args.max_hours,
        settings=settings.to_dict(),
    )
    print(f"DPO 完成，推理适配器：{final}")


if __name__ == "__main__":
    main()
