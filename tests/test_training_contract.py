"""只检查会改变实验结论的关键算式与输入约束，不运行完整训练。"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_sft import eval_ngrams, overlaps_eval  # noqa: E402
from prepare_preference import python_code  # noqa: E402
from train_dpo import dpo_loss  # noqa: E402
from train_grpo import grpo_loss, group_advantages  # noqa: E402
from train_ppo import action_logprobs, gae, ppo_loss  # noqa: E402


class TrainingContractTest(unittest.TestCase):
    def test_preference_requires_code_only(self):
        self.assertEqual(python_code("\ufeff```python\ndef f(x):\n    return x\n```"), "def f(x):\n    return x\n")
        self.assertIsNone(python_code("Here is a solution:\ndef f(x): return x"))

    def test_eval_overlap(self):
        task = [{"prompt": "Write a Python function to reverse every word in a sentence."}]
        spans = eval_ngrams(task)
        self.assertTrue(overlaps_eval(task[0]["prompt"], spans))
        self.assertFalse(overlaps_eval("Implement a binary search tree insertion", spans))

    def test_ppo_token_alignment_and_advantage(self):
        ids = torch.tensor([[0, 1, 2, 3]])  # 前 2 个是题目，后 2 个是模型动作。
        logits = torch.zeros(1, 4, 5)
        logits[0, 1, 2] = 10  # 位置 1 预测第一个答案 token 2。
        logits[0, 2, 3] = 10  # 位置 2 预测第二个答案 token 3。
        self.assertEqual(action_logprobs(logits, ids, 2).shape, (2,))
        self.assertTrue(torch.all(action_logprobs(logits, ids, 2) > -0.001))
        self.assertTrue(torch.all(action_logprobs(logits, ids, 2, temperature=0.8) > -0.001))
        advantages, returns = gae(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0]), lam=1.0)
        self.assertTrue(torch.allclose(advantages, torch.tensor([1.0, 1.0])))
        self.assertTrue(torch.allclose(returns, torch.tensor([1.0, 1.0])))
        total, policy, value = ppo_loss(
            torch.zeros(2), torch.zeros(2), torch.zeros(2), torch.zeros(2),
            advantages, returns,
        )
        self.assertTrue(torch.isfinite(total))
        self.assertAlmostEqual(policy.item(), -1.0)
        self.assertAlmostEqual(value.item(), 0.5)

    def test_dpo_prefers_chosen_relative_to_frozen_reference(self):
        # 两个模型起点相同时，优势差为 0，DPO 损失应为 ln(2)。
        chosen = torch.tensor(-2.0, requires_grad=True)
        rejected = torch.tensor(-2.0, requires_grad=True)
        loss = dpo_loss(chosen, rejected, torch.tensor(-2.0), torch.tensor(-2.0))
        self.assertAlmostEqual(loss.item(), 0.693147, places=5)
        loss.backward()
        self.assertLess(chosen.grad.item(), 0)  # 梯度下降会提高 chosen 的 logp。
        self.assertGreater(rejected.grad.item(), 0)  # 梯度下降会降低 rejected 的 logp。

    def test_grpo_group_advantage_and_clipped_loss(self):
        advantages = group_advantages([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(advantages.mean().item(), 0.0)
        self.assertGreater(advantages[0].item(), 0)
        self.assertTrue(torch.all(advantages[1:] < 0))
        self.assertTrue(torch.equal(group_advantages([0.0] * 4), torch.zeros(4)))
        # 正优势的答案已比旧策略提高很多时，clip 后不再继续推动它。
        new = torch.tensor([1.0, 1.0], requires_grad=True)
        loss = grpo_loss(new, torch.zeros(2), torch.tensor(1.0))
        loss.backward()
        self.assertTrue(torch.equal(new.grad, torch.zeros(2)))


if __name__ == "__main__":
    unittest.main()
