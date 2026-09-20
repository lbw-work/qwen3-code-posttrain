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
