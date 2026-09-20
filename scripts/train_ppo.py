"""教学版单卡 RLHF：冻结奖励模型/参考策略，训练 LoRA 策略与价值头。"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from generate_eval import ROOT, sha256
from train_common import locked_jsonl, new_run, save_meta, sft_adapter


def action_logprobs(logits: torch.Tensor, ids: torch.Tensor, prompt_length: int) -> torch.Tensor:
    """返回每个新 token 的 logπ(a_t|s_t)，形状 [生成 token 数]。"""
    # logits[0, j] 预测 ids[0, j+1]。因此第一个生成 token
    # ids[0, prompt_length] 对应 logits[0, prompt_length-1]。
    selected = logits[0, prompt_length - 1:-1].float().log_softmax(-1)
    actions = ids[0, prompt_length:].unsqueeze(-1)
    return selected.gather(-1, actions).squeeze(-1)


def gae(rewards: torch.Tensor, values: torch.Tensor, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    """从最后一个 token 向前递推优势 A_t，最后状态的 bootstrap 值取 0。"""
    advantage = torch.zeros_like(rewards)
    running = rewards.new_zeros(())
    for index in range(len(rewards) - 1, -1, -1):
        next_value = values[index + 1] if index + 1 < len(values) else values.new_zeros(())
        delta = rewards[index] + next_value - values[index]  # gamma=1
        running = delta + lam * running
        advantage[index] = running
    return advantage, advantage + values


def ppo_loss(
    new_logp: torch.Tensor, old_logp: torch.Tensor, new_values: torch.Tensor,
    old_values: torch.Tensor, advantages: torch.Tensor, returns: torch.Tensor,
    clip: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPO clipped policy loss + clipped value loss；仅在本次 rollout 的 token 上计算。"""
    ratio = (new_logp - old_logp).clamp(-20, 20).exp()
    policy = -torch.minimum(
        ratio * advantages, ratio.clamp(1 - clip, 1 + clip) * advantages,
    ).mean()
    clipped_values = old_values + (new_values - old_values).clamp(-clip, clip)
    value = 0.5 * torch.maximum(
        (new_values - returns).square(), (clipped_values - returns).square(),
    ).mean()
    return policy + 0.5 * value, policy.detach(), value.detach()


def load_policy(adapter: Path, trainable: bool):
    base = AutoModelForCausalLM.from_pretrained(
        ROOT / "models" / "base", dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )
    return PeftModel.from_pretrained(base, adapter, is_trainable=trainable).to("cuda")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-run", required=True)
    parser.add_argument("--reward-run", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-hours", type=float, default=2)
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_hours <= 0:
        parser.error("--max-steps 和 --max-hours 必须为正数")
    if torch.cuda.device_count() != 1:
        parser.error("PPO 正式训练要求一张 CUDA 显卡")
    sft = sft_adapter(ROOT / args.sft_run)
    reward_run = (ROOT / args.reward_run).resolve()
    if not reward_run.is_relative_to((ROOT / "runs" / "reward").resolve()):
        parser.error("--reward-run 必须是 runs/reward/ 下的一次训练")
    reward_adapter = reward_run / "final_model"
    if not (reward_adapter / "adapter_config.json").is_file():
        parser.error("奖励模型适配器不存在")
    train_file, _, lock = locked_jsonl(ROOT / "data" / "rl", "rl")
    prompts = load_dataset("json", data_files=str(train_file), split="train")["prompt"]
    if not prompts:
        parser.error("RL 提示词数据为空")
    run = new_run("ppo", args.run_id)
    rng = random.Random(42)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(sft, local_files_only=True)

    policy = load_policy(sft, trainable=True)
    reference = load_policy(sft, trainable=False).eval()
    # PPO 的 old_logp 与 new_logp 必须基于同一确定网络；训练时关闭 LoRA dropout。
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    reward_base = AutoModelForSequenceClassification.from_pretrained(
        ROOT / "models" / "base", num_labels=1, dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )
    reward_base.config.pad_token_id = tokenizer.pad_token_id
    reward_model = PeftModel.from_pretrained(reward_base, reward_adapter, is_trainable=False).to("cuda").eval()
    for parameter in reward_model.parameters():
        parameter.requires_grad_(False)

    # 价值头读取 policy 最后一层每个位置的隐藏状态 [T,H]，输出 [T]。
    # 它估计“写下下一个 token 前，后续预期能拿到多少奖励”。
    value_head = torch.nn.Linear(policy.config.hidden_size, 1).to("cuda", dtype=torch.float32)
    torch.nn.init.zeros_(value_head.weight)
    torch.nn.init.zeros_(value_head.bias)
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad] + list(value_head.parameters()), lr=5e-6,
    )
    policy.config.use_cache = False
    started = time.monotonic()
    history = run / "train_log.jsonl"
    completed_steps = 0

    with history.open("w", encoding="utf-8") as log:
        for step in range(1, args.max_steps + 1):
            if time.monotonic() - started >= args.max_hours * 3600:
                break
            question = rng.choice(prompts)
            prompt_ids = tokenizer(
                question, return_tensors="pt", truncation=True, max_length=512,
                add_special_tokens=False,
            )["input_ids"].to("cuda")
            prompt_length = prompt_ids.shape[1]
            policy.eval()
            with torch.no_grad():
                # rollout 是旧策略采样出来的动作序列。之后多次更新时，
                # old_logp 固定，PPO 的概率比才有明确的分母。
                ids = policy.generate(
                    input_ids=prompt_ids, do_sample=True, temperature=0.8,
                    max_new_tokens=256, pad_token_id=tokenizer.eos_token_id,
                )
                old_output = policy(ids, output_hidden_states=True)
                ref_output = reference(ids)
                old_logp = action_logprobs(old_output.logits, ids, prompt_length).detach()
                ref_logp = action_logprobs(ref_output.logits, ids, prompt_length).detach()
                old_hidden = old_output.hidden_states[-1][0, prompt_length - 1:-1].float()
                old_values = value_head(old_hidden).squeeze(-1).detach()
                rm_score = reward_model(input_ids=ids).logits[0, 0].float().clamp(-5, 5).detach()

            # 每个 token 扣除与冻结 SFT reference 的 log 概率差；
            # 可执行代码的总体奖励只加在最后一个生成 token 上。
            rewards = -0.02 * (old_logp - ref_logp)
            rewards[-1] += rm_score
            advantages, returns = gae(rewards, old_values)
            if len(advantages) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
            advantages, returns = advantages.detach(), returns.detach()

            policy.train()
            for _ in range(2):  # 同一 rollout 最多更新两遍；更多遍更容易偏离采样策略。
                optimizer.zero_grad(set_to_none=True)
                output = policy(ids, output_hidden_states=True)
                new_logp = action_logprobs(output.logits, ids, prompt_length)
                hidden = output.hidden_states[-1][0, prompt_length - 1:-1].float()
                new_values = value_head(hidden).squeeze(-1)
                loss, policy_part, value_part = ppo_loss(
                    new_logp, old_logp, new_values, old_values, advantages, returns,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad] + list(value_head.parameters()), 1.0,
                )
                optimizer.step()
            completed_steps = step
            row = {
                "step": step, "reward_model_score": rm_score.item(),
                "mean_kl": (old_logp - ref_logp).mean().item(),
                "policy_loss": policy_part.item(), "value_loss": value_part.item(),
                "response_tokens": ids.shape[1] - prompt_length,
                "elapsed_seconds": time.monotonic() - started,
                "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            if step % 25 == 0:
                checkpoint = run / f"checkpoint-{step}"
                policy.save_pretrained(checkpoint)
                torch.save(value_head.state_dict(), checkpoint / "value_head.pt")
            print(row, flush=True)

    final = run / "final_model"
    policy.save_pretrained(final)
    tokenizer.save_pretrained(final)
    torch.save(value_head.state_dict(), run / "value_head.pt")
    save_meta(
        run, stage="ppo", sft_run=str(sft.parent), reward_run=str(reward_run),
        sft_adapter_config_sha256=sha256(sft / "adapter_config.json"),
        reward_adapter_config_sha256=sha256(reward_adapter / "adapter_config.json"),
        rl_lock_sha256=sha256(ROOT / "data" / "rl.lock.json"),
        train_sha256=lock["train_sha256"], eval_sha256=lock["eval_sha256"],
        completed_steps=completed_steps, max_hours=args.max_hours,
        elapsed_seconds=time.monotonic() - started, seed=42,
        beta_kl=0.02, ppo_epochs=2, clip=0.2,
    )
    print(f"PPO 完成，推理适配器：{final}")


if __name__ == "__main__":
    main()
