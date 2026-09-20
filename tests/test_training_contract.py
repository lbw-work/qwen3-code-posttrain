"""只检查会改变实验结论的关键算式与输入约束，不运行完整训练。"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_sft import eval_ngrams, overlaps_eval  # noqa: E402
from prepare_preference import python_code  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
