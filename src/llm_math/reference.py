"""Small, deterministic reference implementations used by tests and notebooks."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    shifted = x - x.amax(dim=dim, keepdim=True)
    exp = shifted.exp()
    return exp / exp.sum(dim=dim, keepdim=True)


def manual_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -log_probs.gather(-1, targets.unsqueeze(-1)).mean()


def finite_difference_gradient(fn, x: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    gradient = torch.empty_like(x)
    flat_x = x.reshape(-1)
    flat_gradient = gradient.reshape(-1)
    for index in range(flat_x.numel()):
        original = flat_x[index].item()
        flat_x[index] = original + eps
        upper = fn(x).item()
        flat_x[index] = original - eps
        lower = fn(x).item()
        flat_x[index] = original
        flat_gradient[index] = (upper - lower) / (2 * eps)
    return gradient


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
    if causal:
        mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    weights = stable_softmax(scores, dim=-1)
    return weights @ value, weights


def apply_rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 2:
        raise ValueError("RoPE requires an even feature dimension")
    half = x.shape[-1] // 2
    frequencies = 1.0 / (10000 ** (torch.arange(half, device=x.device) / half))
    angles = positions.to(x.dtype).unsqueeze(-1) * frequencies
    left, right = x[..., :half], x[..., half:]
    return torch.cat(
        [left * angles.cos() - right * angles.sin(), left * angles.sin() + right * angles.cos()],
        dim=-1,
    )


def lora_delta(a: torch.Tensor, b: torch.Tensor, alpha: float) -> torch.Tensor:
    rank = a.shape[0]
    if rank == 0 or b.shape[1] != rank:
        raise ValueError("LoRA factors have incompatible ranks")
    return (b @ a) * (alpha / rank)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    preference_margin = (policy_chosen - policy_rejected) - (
        reference_chosen - reference_rejected
    )
    return -F.logsigmoid(beta * preference_margin).mean()
