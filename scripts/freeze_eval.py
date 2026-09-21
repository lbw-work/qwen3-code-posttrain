"""固定 EvalPlus 题目和测试快照，生成之后所有模型共用的 eval.jsonl。"""

import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_mbpp_plus
from evalplus.data import humaneval, mbpp


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "evalplus"


def sha256(path: Path) -> str:
    """分块计算文件的 SHA256，用内容而非文件名锁定题库。

    ``eval.lock.json`` 保存的是这个返回值。以后即使某个缓存文件仍叫原来的
    名字，只要内容被升级、替换或损坏，哈希值就会不同。每次只读取 1 MiB，
    避免大题库文件被一次性读入内存。
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """把 EvalPlus 的两套公开题目和隐藏测试快照固定为本项目的评测基线。

    该函数只能在一个新实验的最开始运行一次：若 ``eval.jsonl`` 或锁文件已经
    存在就立即报错，防止中途换题而让不同训练阶段的分数失去可比性。它做三件
    事：

    1. 调用 EvalPlus 读取 HumanEval+ 和 MBPP+，并把官方缓存的测试数据复制到
       ``data/evalplus``；测试用例不混入训练数据。
    2. 写出 ``eval.jsonl``。每行只保留生成时需要的 ``prompt``、函数入口名和
       题目 ID；评分脚本再根据锁定快照取隐藏测试，因此模型不会看到答案。
    3. 写出 ``eval.lock.json``，记录 EvalPlus 版本、两套数据版本、题目数及每个
       文件的 SHA256，作为之后所有 Base/SFT/RL 评测的共同契约。

    写清单时先写临时文件，完成后用 ``os.replace`` 原子替换，避免意外中断留下
    半份 ``eval.jsonl``。项目已存在上述文件时，阅读代码即可，不应重新运行它。
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "eval.jsonl"
    if manifest.exists() or (ROOT / "eval.lock.json").exists():
        raise FileExistsError("评测清单已经冻结；请勿在实验中途重新生成")
    lock = {"evalplus_version": importlib.metadata.version("evalplus"), "datasets": {}}
    rows = []

    # EvalPlus 的 Python API 会下载并解析官方题库，但不会执行题目里的代码。
    # 复制其缓存文件，是为了以后升级 EvalPlus 包时仍能使用这次实验的原始测试。
    for suite, load, source, version in (
        ("humaneval", get_human_eval_plus, humaneval, humaneval.HUMANEVAL_PLUS_VERSION),
        ("mbpp", get_mbpp_plus, mbpp, mbpp.MBPP_PLUS_VERSION),
    ):
        tasks = load()
        dataset_name = "HumanEvalPlus" if suite == "humaneval" else "MbppPlus"
        _, cache_file = source.get_dataset_metadata(dataset_name, version, False, False)
        destination = DATA_DIR / Path(cache_file).name
        shutil.copyfile(cache_file, destination)
        lock["datasets"][suite] = {
            "version": version,
            "file": str(destination.relative_to(ROOT)),
            "sha256": sha256(destination),
            "count": len(tasks),
        }

        # 清单只放题干与题目编号。标准答案和测试留在上面的题库快照中，
        # 避免训练数据处理脚本误把评测答案读进 SFT / 偏好数据。
        for task_id, task in tasks.items():
            rows.append({
                "suite": suite,
                "task_id": task_id,
                "prompt": task["prompt"],
                "entry_point": task["entry_point"],
            })

    # 先写临时文件再替换，避免程序中途退出留下不完整的评测清单。
    temporary = manifest.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, manifest)
    lock["eval_jsonl_sha256"] = sha256(manifest)
    (ROOT / "eval.lock.json").write_text(
        json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"固定 {len(rows)} 道题：{manifest}")


if __name__ == "__main__":
    main()
