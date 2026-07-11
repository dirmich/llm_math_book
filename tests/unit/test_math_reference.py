import math

import torch
import torch.nn.functional as F

from llm_math.reference import (
    apply_rope,
    dpo_loss,
    finite_difference_gradient,
    lora_delta,
    manual_cross_entropy,
    scaled_dot_product_attention,
    stable_softmax,
)


def test_softmax_is_stable_and_normalized():
    probabilities = stable_softmax(torch.tensor([[10000.0, 10001.0, 9999.0]]))
    assert torch.isfinite(probabilities).all()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(1))


def test_cross_entropy_matches_pytorch():
    logits = torch.tensor([[1.0, -2.0, 0.5], [0.1, 0.2, 0.3]])
    targets = torch.tensor([2, 0])
    torch.testing.assert_close(manual_cross_entropy(logits, targets), F.cross_entropy(logits, targets))


def test_finite_difference_matches_backpropagation():
    x = torch.tensor([0.3, -0.7], dtype=torch.float64, requires_grad=True)
    loss = (x.sin() + x.square()).sum()
    loss.backward()
    numerical = finite_difference_gradient(lambda value: (value.sin() + value.square()).sum(), x.detach().clone())
    torch.testing.assert_close(numerical, x.grad, rtol=1e-5, atol=1e-6)


def test_attention_uses_sqrt_dk_and_causal_mask():
    torch.manual_seed(0)
    q = torch.randn(1, 4, 8)
    k = torch.randn(1, 4, 8)
    v = torch.randn(1, 4, 8)
    output, weights = scaled_dot_product_attention(q, k, v, causal=True)
    scores = q @ k.transpose(-1, -2) / math.sqrt(8)
    scores = scores.masked_fill(torch.ones(4, 4, dtype=torch.bool).triu(1), float("-inf"))
    expected = F.softmax(scores, dim=-1)
    assert output.shape == q.shape
    torch.testing.assert_close(weights.triu(1), torch.zeros_like(weights))
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, 4))
    torch.testing.assert_close(weights.tril(), expected.tril())


def test_rope_preserves_norm_and_depends_on_position():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    rotated = apply_rope(x, torch.tensor([0, 3]))
    torch.testing.assert_close(rotated.norm(dim=-1), x.norm(dim=-1))
    assert not torch.equal(rotated[0], rotated[1])


def test_lora_initial_delta_and_dpo_preference_direction():
    a = torch.randn(2, 4)
    b = torch.zeros(3, 2)
    torch.testing.assert_close(lora_delta(a, b, alpha=2), torch.zeros(3, 4))
    preferred = dpo_loss(torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
    reversed_order = dpo_loss(torch.tensor([0.0]), torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([0.0]))
    assert preferred < reversed_order
