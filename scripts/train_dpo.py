"""从同一 LoRA SFT 起点手写 DPO 损失和单卡训练循环。"""

import argparse
import json
import math
import random
import shutil
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from generate_eval import ROOT, sha256
from train_common import locked_jsonl, new_run, save_meta, sft_adapter
from train_ppo import action_logprobs


BETA = 0.1
MAX_LENGTH = 1024
ACCUMULATION = 16
EVAL_EVERY = 200


def load_sft_copy(adapter, trainable: bool):
    """从相同的 Base 和 SFT 适配器分别构造 policy、reference。

    policy 的 LoRA 权重接收梯度；reference 的所有权重固定。两份对象必须独立，
    否则 policy 每更新一次，DPO 所比较的“原始 SFT 概率”也会跟着变化。
    """
    base = AutoModelForCausalLM.from_pretrained(
        ROOT / "models" / "base", dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True,
    )
    return PeftModel.from_pretrained(base, adapter, is_trainable=trainable).to("cuda")


def answer_input(tokenizer, prompt: str, answer: str, device: torch.device) -> tuple[torch.Tensor, int]:
    """把一条题目与一份答案编码为 ``[1, prompt + answer + EOS]``。

    返回的第二个数是 prompt 的 token 长度。后面只累加答案和 EOS 的对数概率，
    不让模型通过“更喜欢题干”来降低 DPO 损失。数据准备脚本已检查 1024 长度，
    此处再检查一次，避免训练时默默截去 chosen 或 rejected 的尾部。
    """
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    suffix = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    if not prefix or not suffix or len(prefix) + len(suffix) > MAX_LENGTH:
        raise ValueError("偏好样本为空或超过 DPO 最大长度")
    return torch.tensor([prefix + suffix], device=device), len(prefix)


def answer_logprob(model, ids: torch.Tensor, prompt_length: int) -> torch.Tensor:
    """求一份完整答案的 token 对数概率之和，保留 policy 的梯度。

    语言模型在位置 ``j`` 的 logits 预测位置 ``j+1`` 的 token；共用的
    ``action_logprobs`` 负责这一步错位对齐。结果是标量，长度不同的答案
    仍按标准 DPO 的序列 log probability 求和，不将 prompt token 算进去。
    """
    return action_logprobs(model(ids).logits, ids, prompt_length).sum()


def dpo_loss(
    chosen_policy: torch.Tensor, rejected_policy: torch.Tensor,
    chosen_reference: torch.Tensor, rejected_reference: torch.Tensor,
    beta: float = BETA,
) -> torch.Tensor:
    """计算一个偏好对的 DPO 损失 ``-log sigmoid(beta * 优势差)``。

    policy 优势 = logπ(chosen|prompt) - logπ(rejected|prompt)。reference
    优势同理，但它始终冻结。两者相减后，只有 policy 比原始 SFT 更偏向
    chosen 时，损失才下降。这样训练的是同题两份代码的相对偏好。
    """
    improvement = (chosen_policy - rejected_policy) - (chosen_reference - rejected_reference)
    return -F.logsigmoid(beta * improvement)


def pair_loss(policy, reference, tokenizer, row: dict) -> torch.Tensor:
    """对同一题的两份答案分别前向，返回一个可反传的 DPO 标量损失。"""
    device = next(policy.parameters()).device
    chosen, prompt_len = answer_input(tokenizer, row["prompt"], row["chosen"], device)
    rejected, rejected_prompt_len = answer_input(tokenizer, row["prompt"], row["rejected"], device)
    with torch.no_grad():
        ref_chosen = answer_logprob(reference, chosen, prompt_len)
        ref_rejected = answer_logprob(reference, rejected, rejected_prompt_len)
    pol_chosen = answer_logprob(policy, chosen, prompt_len)
    pol_rejected = answer_logprob(policy, rejected, rejected_prompt_len)
    return dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected)


@torch.no_grad()
def validation_loss(policy, reference, tokenizer, rows) -> float:
    """逐对计算偏好验证集平均损失，用于选择最佳适配器，不使用正式评测题。"""
    policy.eval()
    result = sum(pair_loss(policy, reference, tokenizer, row).item() for row in rows) / len(rows)
    if not math.isfinite(result):
        raise FloatingPointError("DPO 验证损失不是有限数，拒绝保存适配器")
    return result


def main() -> None:
    """手写 DPO：每 16 对样本更新，定期验证，并保存验证损失最低的适配器。

    墙钟限制只在完整梯度更新之后检查；``--max-steps`` 限制更新次数，可在
    4090 上先跑一两步确认显存、loss 和保存路径。日志保留每次更新的损失。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft-run", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--max-hours", type=float, default=4)
    args = parser.parse_args()
    if args.max_steps == 0 or args.max_steps < -1 or args.max_hours <= 0:
        parser.error("--max-steps 必须为 -1 或正整数，--max-hours 必须为正数")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        parser.error("DPO 正式训练要求一张支持 bf16 的 CUDA 显卡")

    adapter = sft_adapter(ROOT / args.sft_run)
    train_file, valid_file, lock = locked_jsonl(ROOT / "data" / "preference", "preference")
    data = load_dataset("json", data_files={"train": str(train_file), "valid": str(valid_file)})
    train_rows, valid_rows = data["train"], data["valid"]
    if not train_rows or not valid_rows:
        parser.error("偏好训练集和验证集都必须非空")
    run = new_run("dpo", args.run_id)
    torch.manual_seed(42)
    order = list(range(len(train_rows)))
    random.Random(42).shuffle(order)
    tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
    policy = load_sft_copy(adapter, trainable=True)
    reference = load_sft_copy(adapter, trainable=False).eval().requires_grad_(False)
    # chosen/rejected 的概率必须可重复；训练态 LoRA dropout 也关闭。
    for module in policy.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    policy.config.use_cache = False
    reference.config.use_cache = False
    policy.gradient_checkpointing_enable()
    policy.enable_input_require_grads()
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=5e-6)

    started = time.monotonic()
    maximum = math.ceil(len(order) / ACCUMULATION) if args.max_steps == -1 else min(args.max_steps, math.ceil(len(order) / ACCUMULATION))
    best_loss = float("inf")
    best_path = run / "checkpoints" / "best"
    completed_steps = 0
    last_validated_step = 0
    with (run / "train_log.jsonl").open("w", encoding="utf-8") as log:
        for start in range(0, len(order), ACCUMULATION):
            if completed_steps >= maximum or time.monotonic() - started >= args.max_hours * 3600:
                break
            group = order[start:start + ACCUMULATION]
            optimizer.zero_grad(set_to_none=True)
            policy.train()
            losses = []
            for index in group:
                loss = pair_loss(policy, reference, tokenizer, train_rows[index])
                if not torch.isfinite(loss):
                    raise FloatingPointError("DPO 训练损失不是有限数")
                (loss / len(group)).backward()
                losses.append(loss.detach().item())
            torch.nn.utils.clip_grad_norm_((p for p in policy.parameters() if p.requires_grad), 1.0)
            optimizer.step()
            completed_steps += 1
            record = {"step": completed_steps, "train_loss": sum(losses) / len(losses),
                      "elapsed_seconds": time.monotonic() - started}
            if completed_steps % EVAL_EVERY == 0 or completed_steps == maximum:
                record["valid_loss"] = validation_loss(policy, reference, tokenizer, valid_rows)
                last_validated_step = completed_steps
                if record["valid_loss"] < best_loss:
                    best_loss = record["valid_loss"]
                    policy.save_pretrained(best_path)
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(record, flush=True)

    # 墙钟上限可能在验证间隔之间到达；最后仍评一次当前权重，确保有最佳模型。
    if last_validated_step != completed_steps:
        current_loss = validation_loss(policy, reference, tokenizer, valid_rows)
        if current_loss < best_loss:
            best_loss = current_loss
            policy.save_pretrained(best_path)
    final = run / "final_model"
    shutil.copytree(best_path, final)
    tokenizer.save_pretrained(final)
    save_meta(
        run, stage="dpo", implementation="manual", sft_run=str(adapter.parent),
        sft_adapter_config_sha256=sha256(adapter / "adapter_config.json"),
        preference_lock_sha256=sha256(ROOT / "data" / "preference.lock.json"),
        train_sha256=lock["train_sha256"], valid_sha256=lock["valid_sha256"],
        eval_sha256=lock["eval_sha256"], best_checkpoint=str(best_path),
        best_eval_loss=best_loss, global_step=completed_steps,
        train_runtime_seconds=time.monotonic() - started, max_hours=args.max_hours,
        settings={"beta": BETA, "max_length": MAX_LENGTH, "gradient_accumulation_steps": ACCUMULATION,
                  "learning_rate": 5e-6, "num_train_epochs": 1, "eval_steps": EVAL_EVERY, "seed": 42},
    )
    print(f"DPO 完成，推理适配器：{final}")


if __name__ == "__main__":
    main()
