"""从固定版本的 OpenCodeInstruct 构建可复现的 Python 函数 SFT 数据。"""

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as parquet
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from generate_eval import ROOT, sha256, verified_tasks


REPO = "nvidia/OpenCodeInstruct"
REVISION = "8f3ba5bafe4d6e8db46082cf7ae6741bc370604d"
SOURCE_FILE = "data/train-00000-of-00050.parquet"
MAX_SEQUENCE_TOKENS = 1024
MAX_COMPLETION_TOKENS = 512
CODE_BLOCK = re.compile(r"```(?:python|py)\s*\n([\s\S]*?)\n```\Z")
WORD = re.compile(r"[a-z0-9_]+")


def words(text: str) -> list[str]:
    """只为找重复题而归一化；原始训练文本不经过这种改写。"""
    return WORD.findall(text.casefold())


def eval_ngrams(tasks: list[dict]) -> dict[int, set[tuple[str, ...]]]:
    """长题干取 12 连词；短题干取完整题干，避免短题复制漏检。"""
    result: dict[int, set[tuple[str, ...]]] = {}
    for task in tasks:
        tokens = words(task["prompt"])
        if len(tokens) < 6:
            continue
        size = min(12, len(tokens))
        result.setdefault(size, set()).update(
            tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)
        )
    return result


def overlaps_eval(question: str, ngrams: dict[int, set[tuple[str, ...]]]) -> bool:
    tokens = words(question)
    return any(
        tuple(tokens[i:i + size]) in spans
        for size, spans in ngrams.items()
        for i in range(len(tokens) - size + 1)
    )


def prepare(source: Path, tokenizer, tasks: list[dict], output: Path) -> dict:
    """逐批读取 Parquet；只保留单个完整代码块、测试全过的函数题。"""
    output.mkdir(parents=True, exist_ok=False)
    duplicates = set()
    ngrams = eval_ngrams(tasks)
    counts = Counter()
    train_path, valid_path = output / "train.jsonl", output / "valid.jsonl"
    with train_path.open("w", encoding="utf-8") as train, valid_path.open("w", encoding="utf-8") as valid:
        reader = parquet.ParquetFile(source)
        for batch in reader.iter_batches(batch_size=1024):
            for item in batch.to_pylist():
                counts["source_rows"] += 1
                # 数据卡提供逐条单元测试结果。这里只信任已记录的执行结果，
                # 不在准备数据时执行第三方代码；原测试质量仍需人工抽查。
                try:
                    statuses = json.loads(item["tests_execution_status"])
                except (TypeError, ValueError):
                    statuses = None
                if item["average_test_score"] != 1.0 or not statuses or any(x != "pass" for x in statuses):
                    counts["not_all_tests_passed"] += 1
                    continue

                question = item["input"].strip()
                response = item["output"].strip()
                match = CODE_BLOCK.fullmatch(response)
                if not question or '"""' in question or match is None:
                    counts["format_rejected"] += 1
                    continue
                code = match.group(1).rstrip() + "\n"
                try:
                    syntax = ast.parse(code)
                except SyntaxError:
                    counts["syntax_rejected"] += 1
                    continue
                if not any(isinstance(node, ast.FunctionDef) for node in syntax.body):
                    counts["no_top_level_function"] += 1
                    continue
                if overlaps_eval(question, ngrams):
                    counts["eval_overlap_rejected"] += 1
                    continue

                # 相同题干只留下第一次出现的答案。这样一题不会同时落入训练集
                # 和验证集；也避免重复题反复放大其梯度权重。
                normalized = " ".join(words(question))
                key = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                if key in duplicates:
                    counts["duplicate_prompt"] += 1
                    continue
                # MBPP+ 的题干也是模块级三引号说明，再续写 Python 代码。
                # 提示与答案分开存储，训练时只对答案 token 算 loss。
                prompt = f'"""\n{question}\n"""\n\n'
                prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
                completion_tokens = len(tokenizer.encode(code, add_special_tokens=False)) + 1  # EOS
                if prompt_tokens + completion_tokens > MAX_SEQUENCE_TOKENS or completion_tokens > MAX_COMPLETION_TOKENS:
                    counts["too_long"] += 1
                    continue
                duplicates.add(key)
                row = {
                    "source_id": item["id"],
                    "prompt": prompt,
                    "completion": code,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                }
                # 用题干哈希做稳定划分。源文件顺序变化时，同一道题的归属不变。
                stream = valid if int(key[:8], 16) % 20 == 0 else train
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts["valid" if stream is valid else "train"] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path, help="已有的原始 Parquet；省略则按固定版本下载")
    args = parser.parse_args()
    tasks = verified_tasks()
    model_dir = ROOT / "models" / "base"
    if not model_dir.is_dir():
        parser.error("请先运行 scripts/download_models.py 下载 Base 模型及 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    source = args.source_file or Path(hf_hub_download(
        repo_id=REPO, repo_type="dataset", revision=REVISION, filename=SOURCE_FILE,
        local_dir=ROOT / "data" / "raw" / "opencodeinstruct",
    ))
    source = source.resolve()
    if not source.is_file():
        parser.error(f"数据文件不存在：{source}")
    output = ROOT / "data" / "sft"
    lock_path = ROOT / "data" / "sft.lock.json"
    if output.exists() or lock_path.exists():
        raise FileExistsError("SFT 数据或锁文件已存在；请先检查，不自动覆盖")
    counts = prepare(source, tokenizer, tasks, output)
    lock = {
        "repo": REPO,
        "revision": REVISION,
        "source_file": SOURCE_FILE,
        "source_sha256": sha256(source),
        "eval_sha256": sha256(ROOT / "eval.jsonl"),
        "tokenizer": "Qwen/Qwen3-1.7B-Base",
        "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "decontamination": "reject shared 12-word span; full span for eval prompts of 6-11 words",
        "counts": counts,
        "train_sha256": sha256(output / "train.jsonl"),
        "valid_sha256": sha256(output / "valid.jsonl"),
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
