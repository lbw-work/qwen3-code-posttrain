# Qwen3 1.7B Python 函数后训练

目标：在单张 RTX 4090 上，从 `Qwen/Qwen3-1.7B-Base` 出发，按同一口径比较 Base、Full SFT、LoRA SFT、DPO、Reward Model + PPO、GRPO 的代码通过率。正式长训练由你在服务器运行；每一步都写入独立目录，不覆盖前一阶段。

## 当前状态

- 已实现模型版本锁定、SFT/RL 数据准备、Full SFT、LoRA SFT、DPO、奖励模型、PPO、GRPO、统一生成、Docker 隔离判分和最终对比脚本。
- Base 已在本机完成全部 542 题的生成与 Docker 判分：HumanEval+ 31/164（18.9%），MBPP+ 214/378（56.6%）。完整逐题记录在 `results/base/20260922T010155Z/`；尚未在 4090 上训练。训练脚本的接口和小模型数据流已检查，显存与训练收敛仍需服务器实测。
- 公开评测固定为 HumanEval+ v0.1.10（164 题）和 MBPP+ v0.2.0（378 题）。`eval.jsonl` 与官方测试快照的哈希见 `eval.lock.json`。独立自建复核题尚未加入，先不要把现有结果称为最终实验结论。
- 标准答案直接判分的自检为 HumanEval+ 163/164、MBPP+ 377/378；HumanEval/32 和 Mbpp/255 连题库自带的答案也未通过当前测试。所有模型仍按完整 164/378 题报告，逐题记录保留。

## 数据流

```text
锁定 Base 权重 ──┬── Base 评测
               ├── Full SFT ──────────────────────────── 评测
               └── LoRA SFT ─┬───────────────────────── 评测
                             ├── DPO ────────────────── 评测
                             ├── 奖励模型 → PPO ──────── 评测
                             └── 测试执行奖励 → GRPO ─── 评测
```

后三条策略路线均从**同一个 LoRA SFT 适配器**出发。DPO 和奖励模型共用 `data/preference/`；PPO 和 GRPO 共用 `data/rl/` 的题干。GRPO 奖励来自训练数据自带的测试，绝不读取 EvalPlus 的隐藏测试。

## 环境与数据准备

服务器建议 Python 3.11，先按 CUDA 驱动安装匹配的 PyTorch，再安装：

公开仓库随附已处理的 `data/sft/`、`data/rl/`、对应锁文件及 Base 判分结果。
服务器克隆后无需重新运行 `prepare_sft.py` 或 `prepare_rl.py`；这两个脚本用于从原始数据重新构建，遇到已有数据会停止，以免覆盖冻结版本。
模型权重没有随 GitHub 仓库上传；服务器运行 `download_models.py`，按 `models.lock.json` 的提交号下载 Base 和参考模型。

```bash
python -m pip install -r requirements.txt
docker build -f Dockerfile.eval -t qwen-code-eval:0.3.1 .
python scripts/download_models.py
```

SFT 数据源是固定版本的 [NVIDIA OpenCodeInstruct](https://huggingface.co/datasets/nvidia/OpenCodeInstruct) 的第一份 100,000 行分片。脚本只保留原数据记录的测试全过、单个 Python 代码块、可解析且含顶层函数的样本，并按题干去重、与公开评测题干做连续词重合检查。当前得到训练 26,805 条、验证 1,386 条；`data/sft.lock.json`、`data/rl.lock.json` 保存源文件与输出哈希。连续词检查不能保证发现所有语义近似题，正式报告应如实注明。

随仓库发布的 SFT 和 RL 数据是从 NVIDIA OpenCodeInstruct（[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)）筛选并转换得到的版本；原始仓库、固定提交号、原始分片哈希及转换后的哈希见 `data/sft.lock.json` 和 `data/rl.lock.json`。仓库未包含原始 Parquet 分片。

偏好数据尚未下载。`scripts/export_themis.py` 可从固定提交号的 [Themis-CodePreference](https://huggingface.co/datasets/project-themis/Themis-CodePreference) 导出 Python 功能正确性子集，并把原始类别整数转成可读名称。然后运行：

```bash
python scripts/export_themis.py
python scripts/prepare_preference.py --source data/raw/themis-python-fc.jsonl --source-name project-themis/Themis-CodePreference@7c366b23590cc9ff8d372bb47280fcd474536344
```

第二步会统一成代码题干与代码答案，检查长度、语法、去重和评测重合，再生成 DPO/奖励模型共同使用的训练/验证集与锁文件。导出整套原始数据较大，因此本机尚未运行这两步。

## 各阶段命令

先跑 SFT；以下 `<编号>` 是脚本输出目录中的运行编号：

```bash
python scripts/train_sft.py --mode full
python scripts/train_sft.py --mode lora
```

有 `data/preference/` 后可以跑 DPO 与奖励模型，再跑 PPO；GRPO 只需要前面的 `data/rl/`：

```bash
python scripts/train_dpo.py --sft-run runs/lora_sft/<编号>
python scripts/train_reward.py
python scripts/train_ppo.py --sft-run runs/lora_sft/<编号> --reward-run runs/reward/<编号>
python scripts/train_grpo.py --sft-run runs/lora_sft/<编号>
```

DPO、PPO、GRPO 的训练循环和损失都由本项目的 PyTorch 代码实现，不调用 TRL 训练器：DPO 对同一题的 chosen/rejected 计算 policy 与冻结 reference 的答案概率差；PPO 用奖励模型分数、KL、GAE 和裁剪损失更新 LoRA 与价值头；GRPO 每题采样 4 份代码，用训练题自带测试的通过比例计算组内优势，再做两遍裁剪更新。GRPO 的 `beta=0` 不额外驻留 reference；相同得分的组不更新。SFT 与奖励模型仍使用 TRL。

每次训练独立保存 `final_model/`、`training_meta.json` 和日志；DPO、GRPO 保存中间适配器检查点，PPO 另存价值头。SFT、DPO、奖励模型用各自的验证 loss 选择模型，评测集不参与选择。GRPO 的 `--max-steps` 表示最多采样多少组题目；PPO 的 `--max-steps` 表示最多执行多少条 rollout。训练脚本要求恰好一张 CUDA 卡，数据和显卡检查在创建运行目录前完成。正式单卡 4090 的显存、耗时和收敛仍需服务器小步试跑验证。

## 所有模型用同一口径评测

```bash
python scripts/generate_eval.py --stage base --model models/base
python scripts/score_eval.py results/base/<编号>
python scripts/summarize_eval.py results/base/<编号>
```

Full SFT 用 `--stage full_sft --model runs/full_sft/<编号>/final_model`；LoRA SFT、DPO、PPO、GRPO 分别指向自己的 `final_model/`。LoRA 评测会自动加载本项目锁定的 Base 底座再叠加适配器。参考模型可用 `--stage reference --model models/reference` 单独测，不进入必需的六阶段集合。

固定生成口径：EvalPlus 原始 prompt；贪心解码；每题只生成一次；最多 512 个新 token；直接判模型原始代码，不做会误删辅助函数的自动清洗。只允许完整 164+378 题进入正式汇总。模型生成代码只在禁网、只读根文件系统和资源受限的 Docker 内执行。

最后显式传入六个阶段的结果目录，避免脚本误选“最新一次”：

```bash
python scripts/compare_eval.py \
  results/base/<编号> results/full_sft/<编号> results/lora_sft/<编号> \
  results/dpo/<编号> results/ppo/<编号> results/grpo/<编号>
```

它在 `results/comparisons/<编号>/` 保存表格和完整元数据。每个阶段自己的逐题输出、逐题判分、配置、pass@1、生成耗时与训练日志仍在原目录。DPO、GRPO 默认各有 4 小时墙钟上限；奖励模型和 PPO 各有 2 小时上限。墙钟不是精确 GPU 使用时长，正式比较还要结合显卡监控和训练日志记录实际计算量。

## 本机检查

```bash
python -m unittest discover -s tests -v
python -m compileall -q scripts tests
```

这些检查覆盖训练题干去重、PPO token 对齐和优势计算；小模型初始化另验证了 SFT 标签掩码、DPO/奖励模型/GRPO 的训练器输入。它们不代表真实 4090 训练已经跑通。
