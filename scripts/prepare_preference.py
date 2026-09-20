"""将已导出的 Python 正确性偏好 JSONL 变成 DPO/RM 共用的冻结格式。"""

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from generate_eval import ROOT, sha256, verified_tasks
from prepare_sft import CODE_BLOCK, eval_ngrams, overlaps_eval, words


def python_code(text: str) -> str | None:
    """只接受完整代码或单个 Python 代码块，避免把说明文字训练成答案。"""
    text = text.lstrip("\ufeff").strip()
    block = CODE_BLOCK.fullmatch(text)
    if block:
        text = block.group(1)
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    if not any(isinstance(node, ast.FunctionDef) for node in tree.body):
        return None
    return text.rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="每行含 input/prompt、chosen、rejected 的 JSONL")
    parser.add_argument("--source-name", required=True, help="数据集名字和固定版本，写入锁文件")
    args = parser.parse_args()
    source = args.source.resolve()
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / "base", local_files_only=True)
    ngrams = eval_ngrams(verified_tasks())
    output = ROOT / "data" / "preference"
    lock_path = ROOT / "data" / "preference.lock.json"
    if output.exists() or lock_path.exists():
        raise FileExistsError("偏好数据已冻结，不自动覆盖")
    output.mkdir(parents=True)
    counts = Counter()
    seen = set()
    with source.open(encoding="utf-8") as raw, (output / "train.jsonl").open("w", encoding="utf-8") as train, (output / "valid.jsonl").open("w", encoding="utf-8") as valid:
        for line in raw:
            counts["source_rows"] += 1
            item = json.loads(line)
            # 上游导出时必须给出语言和偏好维度；防止把其它语言、代码风格
            # 或安全性偏好混入“Python 功能正确性”实验。
            if item.get("language", "").casefold() != "python" or item.get("aspect", "").casefold() not in ("functional correctness", "fc"):
                counts["wrong_subset"] += 1
                continue
            question = (item.get("input") or item.get("prompt") or "").strip()
            chosen = python_code(item.get("chosen") or "")
            rejected = python_code(item.get("rejected") or "")
            if not question or '"""' in question or chosen is None or rejected is None or chosen == rejected:
                counts["format_rejected"] += 1
                continue
            if overlaps_eval(question, ngrams):
                counts["eval_overlap_rejected"] += 1
                continue
            key = hashlib.sha256(" ".join(words(question)).encode()).hexdigest()
            if key in seen:
                counts["duplicate_prompt"] += 1
                continue
            prompt = f'"""\n{question}\n"""\n\n'
            prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
            if max(prompt_len + len(tokenizer.encode(answer, add_special_tokens=False)) + 1 for answer in (chosen, rejected)) > 1024:
                counts["too_long"] += 1
                continue
            seen.add(key)
            row = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            stream = valid if int(key[:8], 16) % 20 == 0 else train
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            counts["valid" if stream is valid else "train"] += 1
    if not counts["train"] or not counts["valid"]:
        raise ValueError("偏好数据没有得到非空的训练/验证划分；请检查字段与过滤结果")
    lock = {
        "source_name": args.source_name,
        "source_sha256": sha256(source),
        "eval_sha256": sha256(ROOT / "eval.jsonl"),
        "counts": dict(counts),
        "train_sha256": sha256(output / "train.jsonl"),
        "valid_sha256": sha256(output / "valid.jsonl"),
    }
    lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(lock, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
