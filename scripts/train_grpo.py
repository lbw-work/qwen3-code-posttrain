"""从 LoRA SFT 出发，手写按单元测试奖励更新策略的 GRPO。"""

import argparse
import json
import random
import time

import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from code_reward import score_batch
from generate_eval import ROOT, sha256
from train_common import locked_jsonl, new_run, save_meta, sft_adapter
from train_ppo import action_logprobs


GROUP_SIZE = 4
MAX_PROMPT_TOKENS = 512
MAX_COMPLETION_TOKENS = 256
CLIP = 0.2
UPDATES_PER_GROUP = 2
TEMPERATURE = 0.8


def group_advantages(scores: list[float]) -> torch.Tensor:
    """把同一道题的四个执行分标准化，得到各候选答案的相对优势。

    先减去本组平均分，再除以本组标准差。若四份代码得分完全一样，
    就返回全零：没有组内优劣证据，不能凭空给其中一份答案正梯度。
    """
    values = torch.tensor(scores, dtype=torch.float32)
    if len(values) != GROUP_SIZE or not torch.isfinite(values).all():
        raise ValueError("GRPO 必须收到四个有限的执行分")
    deviation = values - values.mean()
    scale = deviation.square().mean().sqrt()
    return deviation / scale if scale > 1e-8 else torch.zeros_like(values)


def grpo_loss(new_logp: torch.Tensor, old_logp: torch.Tensor, advantage: torch.Tensor) -> torch.Tensor:
    """计算一份答案的 token 级裁剪策略损失。

    ``old_logp`` 是采样时策略给生成 token 的概率，更新期间保持固定；
    ``new_logp`` 来自当前策略，保留梯度。逐 token 的概率比
    ``exp(new-old)`` 被限制在 [0.8, 1.2] 附近，避免同一组数据反复
    更新时一步走太远。把同一个组内优势作用到这份答案的每个生成 token，
    再对 token 取平均，避免长答案天然获得更大的更新权重。
    """
    if new_logp.shape != old_logp.shape or new_logp.ndim != 1:
        raise ValueError("新旧 token 对数概率必须是长度相同的一维向量")
    ratio = (new_logp - old_logp).clamp(-20, 20).exp()
    return -torch.minimum(ratio * advantage, ratio.clamp(1 - CLIP, 1 + CLIP) * advantage).mean()


def main() -> None:
    """对每道题采样四份代码，执行训练测试，再按组内奖励手写 GRPO 更新。

    每个 rollout 随机取一个 RL 训练题，四份候选代码从同一个旧策略采样；
    Docker 只执行该训练题自带的 tests，不读取 EvalPlus。四个旧 log probability
    先固定，然后对相同候选做两遍裁剪更新。模型只训练 LoRA 参数，没有 value
    head；``beta=0`` 表示不加载额外 reference，适应单张 4090 的显存。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-run", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--max-steps", type=int, default=200, help="最多采样多少组题目")
    parser.add_argument("--max-hours", type=float, default=4)
    args = parser.parse_args()
    if args.max_steps < 1 or args.max_hours <= 0:
        parser.error("--max-steps 和 --max-hours 必须为正数")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        parser.error("GRPO 正式训练要求一张支持 bf16 的 CUDA 显卡")
    adapter = sft_adapter(ROOT / args.sft_run)
    train_file, _, lock = locked_jsonl(ROOT / "data" / "rl", "rl")
    rows = load_dataset("json", data_files=str(train_file), split="train")
    if not rows or not isinstance(rows[0]["tests"], list):
        parser.error("RL 训练集为空或 tests 不是字符串数组")

    run = new_run("grpo", args.run_id)
    rng = random.Random(42)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    base = AutoModelForCausalLM.from_pretrained(
        ROOT / "models" / "base", dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )
    policy = PeftModel.from_pretrained(base, adapter, is_trainable=True).to("cuda")
    policy.config.use_cache = False
    policy.gradient_checkpointing_enable()
    policy.enable_input_require_grads()
    # 采样时与反传时必须使用同一确定网络；否则新旧概率的差会混入 dropout 噪声。
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-6)
    started = time.monotonic()
    completed_groups = 0
    optimizer_steps = 0
    zero_variance_groups = 0
    with (run / "train_log.jsonl").open("w", encoding="utf-8") as log:
        for step in range(1, args.max_steps + 1):
            if time.monotonic() - started >= args.max_hours * 3600:
                break
            row = rows[rng.randrange(len(rows))]
            prompt_ids = tokenizer(
                row["prompt"], return_tensors="pt", truncation=True,
                max_length=MAX_PROMPT_TOKENS, add_special_tokens=False,
            )["input_ids"].to("cuda")
            if prompt_ids.shape[1] == 0:
                raise ValueError("RL 题干 token 为空")
            prompt_length = prompt_ids.shape[1]
            policy.eval()
            rollouts = []
            completions = []
            with torch.no_grad():
                for _ in range(GROUP_SIZE):
                    # 关闭 top-k 截断，旧/新 logp 才能与实际采样的温度分布一致。
                    ids = policy.generate(
                        input_ids=prompt_ids, do_sample=True, temperature=TEMPERATURE,
                        top_k=0, top_p=1.0,
                        max_new_tokens=MAX_COMPLETION_TOKENS,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    generated = ids[0, prompt_length:]
                    completions.append(tokenizer.decode(generated, skip_special_tokens=True))
                    old_logp = action_logprobs(policy(ids).logits, ids, prompt_length, TEMPERATURE).detach()
                    rollouts.append((ids, old_logp))
            scores = score_batch(completions, [row["tests"]] * GROUP_SIZE)
            advantages = group_advantages(scores).to("cuda")
            completed_groups = step
            if torch.count_nonzero(advantages) == 0:
                zero_variance_groups += 1
            else:
                policy.train()
                for _ in range(UPDATES_PER_GROUP):
                    optimizer.zero_grad(set_to_none=True)
                    for (ids, old_logp), advantage in zip(rollouts, advantages, strict=True):
                        new_logp = action_logprobs(policy(ids).logits, ids, prompt_length, TEMPERATURE)
                        loss = grpo_loss(new_logp, old_logp, advantage) / GROUP_SIZE
                        if not torch.isfinite(loss):
                            raise FloatingPointError("GRPO 训练损失不是有限数")
                        loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    optimizer_steps += 1
            record = {
                "group": step, "rewards": scores, "advantages": advantages.tolist(),
                "optimizer_steps": optimizer_steps, "zero_variance_groups": zero_variance_groups,
                "elapsed_seconds": time.monotonic() - started,
                "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            if step % 50 == 0:
                policy.save_pretrained(run / "checkpoints" / f"group-{step}")
            print(record, flush=True)

    final = run / "final_model"
    policy.save_pretrained(final)
    tokenizer.save_pretrained(final)
    save_meta(
        run, stage="grpo", implementation="manual", sft_run=str(adapter.parent),
        sft_adapter_config_sha256=sha256(adapter / "adapter_config.json"),
        rl_lock_sha256=sha256(ROOT / "data" / "rl.lock.json"),
        train_sha256=lock["train_sha256"], valid_sha256=lock["valid_sha256"],
        eval_sha256=lock["eval_sha256"], completed_groups=completed_groups,
        optimizer_steps=optimizer_steps, zero_variance_groups=zero_variance_groups,
        train_runtime_seconds=time.monotonic() - started, max_hours=args.max_hours,
        settings={"group_size": GROUP_SIZE, "max_prompt_tokens": MAX_PROMPT_TOKENS,
                  "max_completion_tokens": MAX_COMPLETION_TOKENS, "clip": CLIP,
                  "updates_per_group": UPDATES_PER_GROUP, "temperature": TEMPERATURE,
                  "top_k": 0, "top_p": 1.0,
                  "learning_rate": 5e-6, "beta": 0.0, "seed": 42},
    )
    print(f"GRPO 完成，推理适配器：{final}")


if __name__ == "__main__":
    main()
