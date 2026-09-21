"""从 LoRA SFT 出发，用代码单元测试的通过比例做 GRPO 奖励。"""

import argparse

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from code_reward import score_batch
from generate_eval import ROOT, sha256
from train_common import TimeBudget, locked_jsonl, new_run, save_meta, sft_adapter


def execution_reward(completions: list[str], tests: list[list[str]], **kwargs) -> list[float]:
    """把 TRL 生成的多份代码交给 Docker 测试，返回与 completion 等长的 [0,1] 分数。

    GRPO 会把一个 prompt 复制 ``num_generations`` 次，因此这里的 tests 已按同样顺序重复。
    返回第 i 个分数必须对应第 i 段代码；score_batch 用“通过测试数 / 总测试数”作为密集
    奖励，随后 GRPO 在同一道题的四个候选答案内部做相对标准化。
    """
    return score_batch(completions, tests)


def main() -> None:
    """从同一 LoRA SFT 起点运行基于可执行测试奖励的 GRPO。

    每个训练 batch 有 4 个 prompt，每个 prompt 采样 4 份最多 256 token 的代码；Docker
    对每份代码运行该训练样本自己的测试。GRPO 不训练 value head，而是比较同一题四个候选
    的相对奖励，鼓励组内高分代码。``beta=0`` 是单卡显存取舍：不额外驻留 reference；这
    不改变所有路线共享 SFT 起点的实验约束。EvalPlus 测试从不进入此奖励函数。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-run", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-hours", type=float, default=4)
    args = parser.parse_args()
    if args.max_hours <= 0:
        parser.error("--max-hours 必须大于 0")
    if torch.cuda.device_count() != 1:
        parser.error("GRPO 正式训练要求一张 CUDA 显卡")
    adapter = sft_adapter(ROOT / args.sft_run)
    train_file, valid_file, lock = locked_jsonl(ROOT / "data" / "rl", "rl")
    data = load_dataset("json", data_files={"train": str(train_file), "valid": str(valid_file)})
    for split in ("train", "valid"):
        data[split] = data[split].select_columns(["prompt", "tests"])
    # 防止 tests 被 JSONL 保存成字符串后，奖励函数把它当成字符列表。
    if not isinstance(data["train"][0]["tests"], list):
        raise ValueError("RL 数据的 tests 字段必须是字符串数组")
    run = new_run("grpo", args.run_id)
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        ROOT / "models" / "base", dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )
    policy = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    settings = GRPOConfig(
        output_dir=str(run / "checkpoints"),
        max_steps=args.max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_completion_length=256,
        temperature=0.8,
        beta=0.0,
        learning_rate=5e-6,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        seed=42,
    )
    # 每题采样四份代码，执行同一组测试。GRPO 以组内相对奖励更新策略：
    # 分数高于同组平均值的代码提高概率，低于平均值的代码降低概率。
    # beta=0 表示不额外加载 reference，减少单卡显存；仍从相同 SFT 起点出发。
    trainer = GRPOTrainer(
        model=policy, reward_funcs=execution_reward, args=settings,
        train_dataset=data["train"], processing_class=tokenizer,
        callbacks=[TimeBudget(args.max_hours)],
    )
    outcome = trainer.train()
    final = run / "final_model"
    trainer.save_model(str(final))
    tokenizer.save_pretrained(final)
    save_meta(
        run, stage="grpo", sft_run=str(adapter.parent),
        sft_adapter_config_sha256=sha256(adapter / "adapter_config.json"),
        rl_lock_sha256=sha256(ROOT / "data" / "rl.lock.json"),
        train_sha256=lock["train_sha256"], valid_sha256=lock["valid_sha256"],
        eval_sha256=lock["eval_sha256"],
        training_loss=outcome.training_loss, global_step=trainer.state.global_step,
        train_runtime_seconds=outcome.metrics.get("train_runtime"),
        max_hours=args.max_hours,
        settings=settings.to_dict(),
    )
    print(f"GRPO 完成，推理适配器：{final}")


if __name__ == "__main__":
    main()
