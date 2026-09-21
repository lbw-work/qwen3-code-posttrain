"""按不可变的 Hugging Face 提交号下载 Base 和官方后训练参考模型。"""

import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "base": "Qwen/Qwen3-1.7B-Base",
    "reference": "Qwen/Qwen3-1.7B",
}


def main() -> None:
    """下载并校验 Base 与参考模型的完整本地快照。

    第一次运行时，函数先向 Hugging Face 查询每个模型当前 ``main`` 指向的
    Git 提交号，再把这个提交号写进仓库根目录的 ``models.lock.json``。从第二次
    开始，无论是在本机还是 4090 服务器，都会只使用这个锁定提交号下载。因此
    ``config.json``、Tokenizer 和权重始终来自同一次发布，实验结果能被复现。

    每个模型下载到 ``models/<名称>`` 后，本函数会确认三个推理必需的配置文件
    都存在，并确认至少有一个 ``.safetensors`` 权重文件。Qwen 的小模型可能是
    单个权重文件，大模型可能拆成多个分片；所以检查“至少一个”而不假设具体
    文件名。成功后打印模型仓库和精确提交号，供实验记录引用。
    """
    lock_path = ROOT / "models.lock.json"
    locked = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else None
    api = HfApi()
    saved = {}
    for name, repo_id in MODELS.items():
        # 第一次解析 main 为提交号；以后（包括在 4090 服务器上）只下载锁定的提交。
        # 这样权重、Tokenizer、Config 均来自同一次模型发布状态。
        if locked:
            if locked[name]["repo_id"] != repo_id:
                raise ValueError(f"模型仓库与锁文件不一致：{name}")
            revision = locked[name]["revision"]
        else:
            revision = api.model_info(repo_id).sha
        target = ROOT / "models" / name
        snapshot_download(repo_id=repo_id, revision=revision, local_dir=target)
        required = ["config.json", "tokenizer.json", "tokenizer_config.json"]
        missing = [file for file in required if not (target / file).is_file()]
        # Base 是单个 safetensors，官方后训练版是分片权重；两种布局都合法。
        if not list(target.glob("*.safetensors")):
            missing.append("*.safetensors")
        if missing:
            raise RuntimeError(f"{repo_id} 缺少文件：{missing}")
        saved[name] = {"repo_id": repo_id, "revision": revision, "path": str(target.relative_to(ROOT))}
        print(f"已下载 {repo_id} @ {revision}")

    if not locked:
        lock_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
