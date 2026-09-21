"""用代码偏好对训练独立的标量奖励模型，供 PPO 给生成结果打分。"""

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import RewardConfig, RewardTrainer

from generate_eval import ROOT, sha256
from train_common import TimeBudget, locked_jsonl, new_run, save_meta


def main() -> None:
    """从冻结偏好对训练一个给 PPO 用的标量奖励模型。

    模型输入是 ``prompt + answer`` 的 token 序列，输出单个 score。RewardTrainer 对同一题
    的 chosen/rejected 分别前向，优化 ``-log(sigmoid(r_chosen-r_rejected))``，只要求好答案
    比坏答案分更高。Base 主干通过 LoRA 训练，额外的 ``score`` 线性头由 modules_to_save
    显式保存；否则下次加载会只剩适配器而丢掉打分头。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-hours", type=float, default=2)
    args = parser.parse_args()
    if args.max_hours <= 0:
        parser.error("--max-hours 必须大于 0")
    if torch.cuda.device_count() != 1:
        parser.error("奖励模型正式训练要求一张 CUDA 显卡")
    train_file, valid_file, lock = locked_jsonl(ROOT / "data" / "preference", "preference")
    data = load_dataset("json", data_files={"train": str(train_file), "valid": str(valid_file)})
    for split in ("train", "valid"):
        data[split] = data[split].select_columns(["prompt", "chosen", "rejected"])
    run = new_run("reward", args.run_id)
    base = ROOT / "models" / "base"
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=1, dtype=torch.bfloat16, attn_implementation="sdpa",
        local_files_only=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    # 奖励模型输出一个标量，而不是下一 token 概率。
    # score 线性层必须和 LoRA 一起训练、一起保存，否则重新加载后打分无意义。
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear", modules_to_save=["score"], task_type="SEQ_CLS",
    )
    settings = RewardConfig(
        output_dir=str(run / "checkpoints"),
        num_train_epochs=1,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        gradient_checkpointing=True,
        bf16=True,
        learning_rate=1e-4,
        max_length=1024,
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
    # RewardTrainer 对同一题计算 r(chosen) 和 r(rejected)，
    # 使用 -log(sigmoid(r(chosen)-r(rejected)))：只要求好答案分数更高，
    # 分数本身不是代码通过率，也不能代替最终 EvalPlus 测试。
    trainer = RewardTrainer(
        model=model, args=settings, train_dataset=data["train"],
        eval_dataset=data["valid"], processing_class=tokenizer, peft_config=lora,
        callbacks=[TimeBudget(args.max_hours)],
    )
    outcome = trainer.train()
    final = run / "final_model"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    save_meta(
        run, stage="reward",
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
    print(f"奖励模型完成：{final}")


if __name__ == "__main__":
    main()
