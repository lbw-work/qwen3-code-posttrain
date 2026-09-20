"""让一个模型逐题生成代码；只生成，不执行模型输出。"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("base", "reference", "full_sft", "lora_sft", "dpo", "ppo", "grpo")
MAX_NEW_TOKENS = 512


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verified_tasks() -> list[dict]:
    """每次推理先验哈希；有人改过题目时立即停止，避免成绩混在一起。"""
    lock = json.loads((ROOT / "eval.lock.json").read_text(encoding="utf-8"))
    manifest = ROOT / "eval.jsonl"
    if sha256(manifest) != lock["eval_jsonl_sha256"]:
        raise ValueError("eval.jsonl 与冻结时的哈希不一致")
    for dataset in lock["datasets"].values():
        if sha256(ROOT / dataset["file"]) != dataset["sha256"]:
            raise ValueError(f"评测题库被修改：{dataset['file']}")
    with manifest.open(encoding="utf-8") as stream:
        tasks = [json.loads(line) for line in stream]
    if len(tasks) != sum(item["count"] for item in lock["datasets"].values()):
        raise ValueError("评测题数与锁文件不一致")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--model", type=Path, required=True, help="完整模型目录")
    parser.add_argument("--limit", type=int, help="每套题只取前 N 题，用于冒烟测试")
    parser.add_argument("--run-id", help="结果目录名；默认使用 UTC 时间")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit 必须大于 0")
    if not args.model.is_dir():
        parser.error(f"模型目录不存在：{args.model}")

    tasks = verified_tasks()
    model_source = None
    model_lock_path = ROOT / "models.lock.json"
    if model_lock_path.exists() and args.stage in ("base", "reference"):
        model_source = json.loads(model_lock_path.read_text(encoding="utf-8"))[args.stage]
        if args.model.resolve() != (ROOT / model_source["path"]).resolve():
            raise ValueError("模型目录与 models.lock.json 记录的目录不一致")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    category = "smoke" if args.limit else args.stage
    run_dir = ROOT / "results" / category / run_id
    # exist_ok=False 避免重复运行时覆盖以前的逐题输出和评测结果。
    run_dir.mkdir(parents=True, exist_ok=False)

    # Base 模型原本是续写模型，因此所有阶段统一输入 EvalPlus 的原始代码题干。
    # 训练后模型也用同一题干、同一贪心解码参数；这样成绩差异不会来自提示词变化。
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    adapter = (args.model / "adapter_config.json").exists()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if adapter:
        # LoRA 保存的只是低秩增量，不是一套完整权重。
        # 明确从本项目锁定的 Base 底座加载，再把该阶段适配器叠上去。
        base = AutoModelForCausalLM.from_pretrained(ROOT / "models" / "base", dtype="auto", local_files_only=True)
        model = PeftModel.from_pretrained(base, args.model, is_trainable=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype="auto", local_files_only=True)
    model.to(device).eval()
    config = {
        "stage": args.stage,
        "model": str(args.model.resolve()),
        "eval_sha256": sha256(ROOT / "eval.jsonl"),
        "prompt_format": "EvalPlus raw prompt; plain code completion",
        "generation": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
        "device": device,
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "python": platform.python_version(),
        "limit_per_suite": args.limit,
        "adapter": adapter,
    }
    if model_source is not None:
        config["model_source"] = model_source
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    started = time.monotonic()
    for suite in ("humaneval", "mbpp"):
        suite_tasks = [task for task in tasks if task["suite"] == suite]
        if args.limit:
            suite_tasks = suite_tasks[: args.limit]
        output = run_dir / f"{suite}.samples.jsonl"
        with output.open("w", encoding="utf-8") as stream:
            for index, task in enumerate(suite_tasks, 1):
                # input_ids 形状是 [1, L]：1 个题目，L 个输入 token。
                # 本实验一次只送一道题，方便把每次生成与 task_id 一一对应。
                inputs = tokenizer(task["prompt"], return_tensors="pt").to(device)
                with torch.inference_mode():
                    tokens = model.generate(
                        **inputs,
                        do_sample=False,
                        max_new_tokens=MAX_NEW_TOKENS,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                # generate 返回形状 [1, L + N] 的张量，前 L 个 token 是原题干，
                # 后 N 个才是模型新写的代码。只截取后半段，避免题干重复。
                generated = tokens[0, inputs["input_ids"].shape[1]:]
                completion = tokenizer.decode(generated, skip_special_tokens=True)
                # EvalPlus 的 solution 字段需要完整代码；原始生成内容也单独保存，
                # 判分直接使用原始代码；单独保留 completion 便于排查失败题。
                row = {
                    "task_id": task["task_id"],
                    "solution": task["prompt"] + completion,
                    "raw_completion": completion,
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                print(f"{suite} {index}/{len(suite_tasks)} {task['task_id']}", flush=True)
    timing = {
        "generation_elapsed_seconds": time.monotonic() - started,
        "cuda_peak_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
    }
    (run_dir / "generation_timing.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
    print(f"逐题输出保存在 {run_dir}")


if __name__ == "__main__":
    main()
