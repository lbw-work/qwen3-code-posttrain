# 数据格式样例

本目录只用于人工阅读数据格式。每个 `*.sample.jsonl` 恰有 10 行，每行都是来源
数据集的一条完整、未截断 JSON 记录。抽样种子、来源文件和 SHA-256 均记录在
[`samples.lock.json`](samples.lock.json)；重新执行下方脚本会得到同一批样例。

```bash
python scripts/sample_datasets.py
```

## 文件和阶段的关系

| 样例文件 | 记录字段 | 被哪些阶段使用 |
| --- | --- | --- |
| `evaluation.sample.jsonl` | `suite`、`task_id`、`prompt`、`entry_point` | Base 与全部训练后模型的统一评测 |
| `sft.sample.jsonl` | `source_id`、`prompt`、`completion`、两个 token 数 | Full SFT、LoRA SFT |
| `rl.sample.jsonl` | `source_id`、`prompt`、`tests` | PPO、GRPO |
| `preference.sample.jsonl` | `prompt`、`chosen`、`rejected` | DPO、Reward Model |

最后一项在当前项目中尚未生成：它必须由 `export_themis.py` 导出固定版本的 Themis，
再由 `prepare_preference.py` 过滤和冻结。不要用其他三类样例代替它，因为 DPO 和
Reward Model 需要同一道题的优劣代码对，而 SFT/RL 数据不具备这个字段。

训练集与验证集的字段结构相同。本目录抽取的是训练集，目的是让你先看清模型实际
读取的 JSON 形状；验证集只用于训练阶段选择 checkpoint，不参与正式 EvalPlus 分数。
