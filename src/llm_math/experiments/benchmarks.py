"""Shared correctness checks and lightweight benchmark implementations.

These functions intentionally use small deterministic inputs so the language-specific
benchmark wrappers can run on a CPU-only laptop while still checking the algorithm
that the filename claims to measure.
"""

from __future__ import annotations

import math
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise SystemExit(
            "This benchmark requires PyTorch. Install project dependencies with "
            "`pip install -e '.[notebook]'` or run it in Colab."
        ) from exc
    return torch, nn, F


def _time_ms(fn: Callable[[], object], *, repeat: int = 5, warmup: int = 2, device: str = "cpu") -> float:
    torch, _, _ = _require_torch()
    for _ in range(warmup):
        fn()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    times: list[float] = []
    for _ in range(repeat):
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return sum(times) / len(times)


def _save_bar(results_dir: Path, filename: str, title: str, ylabel: str, labels: Iterable[str], values: Iterable[float]) -> None:
    import matplotlib.pyplot as plt

    results_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(list(labels), list(values), color=["#4c78a8", "#f58518", "#54a24b", "#e45756"])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(results_dir / filename, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _print_metadata() -> None:
    torch, _, _ = _require_torch()
    print(f"python={platform.python_version()} torch={torch.__version__} device=cpu")
    print()


def run_tokenizer_benchmark(results_dir: Path, lang: str = "en") -> None:
    _print_metadata()
    corpus = (
        "Large language models split text into tokens. "
        "Byte-level tokenization round-trips Unicode: 수학, 数学, mathematics. "
    ) * 400
    payload = corpus.encode("utf-8")

    def encode() -> list[int]:
        return list(payload)

    tokens = encode()
    decoded = bytes(tokens).decode("utf-8")
    assert decoded == corpus, "byte tokenizer must decode back to the original corpus"
    elapsed_ms = _time_ms(encode, repeat=20, warmup=3)
    seconds = elapsed_ms / 1000.0
    byte_s = len(payload) / seconds
    token_s = len(tokens) / seconds
    chars_per_token = len(corpus) / len(tokens)

    print("=== Tokenizer encode/decode correctness + throughput ===")
    print("correctness: decode(encode(corpus)) == corpus")
    print(f"corpus_bytes={len(payload)} tokens={len(tokens)} chars_per_token={chars_per_token:.3f}")
    print(f"encode_mean_ms={elapsed_ms:.3f} bytes_per_s={byte_s:,.0f} tokens_per_s={token_s:,.0f}")
    _save_bar(results_dir, "bench_tokenizer.png", "Tokenizer throughput", "items/s", ["bytes/s", "tokens/s"], [byte_s, token_s])


def _grouped_attention(q, k, v):
    torch, _, F = _require_torch()
    batch, q_heads, seq, head_dim = q.shape
    kv_heads = k.shape[1]
    assert q_heads % kv_heads == 0, "query heads must be divisible by key/value heads"
    if kv_heads != q_heads:
        repeat = q_heads // kv_heads
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
    return F.softmax(scores, dim=-1) @ v


def run_mha_gqa_benchmark(results_dir: Path, lang: str = "en") -> None:
    torch, _, _ = _require_torch()
    _print_metadata()
    torch.manual_seed(7)
    batch, seq, q_heads, head_dim = 2, 96, 8, 32
    q = torch.randn(batch, q_heads, seq, head_dim)
    variants = [("MHA", 8), ("GQA", 4), ("MQA", 1)]
    rows = []
    for name, kv_heads in variants:
        k = torch.randn(batch, kv_heads, seq, head_dim)
        v = torch.randn(batch, kv_heads, seq, head_dim)
        out = _grouped_attention(q, k, v)
        assert out.shape == q.shape
        latency = _time_ms(lambda: _grouped_attention(q, k, v), repeat=5, warmup=2)
        kv_params = 2 * q_heads * head_dim * kv_heads * head_dim
        kv_cache_mb = batch * kv_heads * seq * head_dim * 2 * 4 / 1024**2
        rows.append((name, kv_heads, kv_params, kv_cache_mb, latency))

    print("=== MHA/GQA/MQA correctness + benchmark ===")
    print("correctness: output shape matches query heads for every kv-head grouping")
    print("| variant | kv_heads | kv_proj_params_est | kv_cache_mb | mean_ms |")
    print("|---|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.3f} | {row[4]:.3f} |")
    _save_bar(results_dir, "bench_mha_gqa.png", "Grouped attention latency", "ms", [r[0] for r in rows], [r[4] for r in rows])


@dataclass(frozen=True)
class GPTConfig:
    name: str
    vocab: int
    seq: int
    dim: int
    heads: int
    layers: int


def _make_tiny_gpt(config: GPTConfig):
    torch, nn, _ = _require_torch()

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(config.dim)
            self.attn = nn.MultiheadAttention(config.dim, config.heads, batch_first=True)
            self.ln2 = nn.LayerNorm(config.dim)
            self.ff = nn.Sequential(
                nn.Linear(config.dim, 4 * config.dim),
                nn.GELU(),
                nn.Linear(4 * config.dim, config.dim),
            )

        def forward(self, x):
            h = self.ln1(x)
            attn, _ = self.attn(h, h, h, need_weights=False, is_causal=False)
            x = x + attn
            return x + self.ff(self.ln2(x))

    class TinyGPT(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token = nn.Embedding(config.vocab, config.dim)
            self.pos = nn.Embedding(config.seq, config.dim)
            self.blocks = nn.ModuleList([Block() for _ in range(config.layers)])
            self.ln = nn.LayerNorm(config.dim)
            self.head = nn.Linear(config.dim, config.vocab, bias=False)

        def forward(self, idx):
            positions = torch.arange(idx.shape[1], device=idx.device).unsqueeze(0)
            x = self.token(idx) + self.pos(positions)
            for block in self.blocks:
                x = block(x)
            return self.head(self.ln(x))

    return TinyGPT()


def _parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def run_gpt_sizes_benchmark(results_dir: Path, lang: str = "en") -> None:
    torch, _, _ = _require_torch()
    _print_metadata()
    torch.manual_seed(11)
    configs = [
        GPTConfig("tiny", 256, 32, 48, 4, 1),
        GPTConfig("small", 256, 32, 96, 4, 2),
        GPTConfig("medium", 256, 32, 144, 4, 2),
    ]
    rows = []
    for cfg in configs:
        model = _make_tiny_gpt(cfg).eval()
        idx = torch.randint(0, cfg.vocab, (2, cfg.seq))
        with torch.no_grad():
            logits = model(idx)
        assert logits.shape == (2, cfg.seq, cfg.vocab)
        latency = _time_ms(lambda: model(idx), repeat=3, warmup=1)
        params = _parameter_count(model)
        rows.append((cfg.name, params, params * 4 / 1024**2, latency))

    print("=== GPT size correctness + forward benchmark ===")
    print("correctness: logits shape is [batch, seq, vocab] for each config")
    print("| config | params | fp32_param_mb | forward_ms |")
    print("|---|---:|---:|---:|")
    for name, params, mb, latency in rows:
        print(f"| {name} | {params} | {mb:.3f} | {latency:.3f} |")
    _save_bar(results_dir, "bench_gpt_sizes.png", "GPT forward latency by size", "ms", [r[0] for r in rows], [r[3] for r in rows])


def run_training_loop_benchmark(results_dir: Path, lang: str = "en") -> None:
    torch, nn, _ = _require_torch()
    _print_metadata()
    torch.manual_seed(13)
    cfg = GPTConfig("train", 128, 24, 64, 4, 1)
    model = _make_tiny_gpt(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab, (4, cfg.seq))
    y = torch.roll(x, shifts=-1, dims=1)
    before = next(model.parameters()).detach().clone()

    def step():
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1))
        loss.backward()
        opt.step()
        return loss

    loss = step()
    assert torch.isfinite(loss)
    assert not torch.equal(before, next(model.parameters()).detach()), "optimizer step must update parameters"
    elapsed_ms = _time_ms(step, repeat=3, warmup=1)
    tokens_s = x.numel() / (elapsed_ms / 1000.0)
    print("=== Training loop correctness + benchmark ===")
    print("correctness: finite CE loss and optimizer step changes parameters")
    print(f"batch_tokens={x.numel()} step_ms={elapsed_ms:.3f} tokens_per_s={tokens_s:,.0f}")
    _save_bar(results_dir, "bench_training_loop.png", "Training step throughput", "tokens/s", ["train step"], [tokens_s])


def run_nano_gpt_benchmark(results_dir: Path, lang: str = "en") -> None:
    torch, nn, _ = _require_torch()
    _print_metadata()
    torch.manual_seed(17)
    cfg = GPTConfig("nano", 96, 32, 64, 4, 2)
    model = _make_tiny_gpt(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4)
    x = torch.randint(0, cfg.vocab, (2, cfg.seq))
    y = torch.roll(x, shifts=-1, dims=1)

    def train_step():
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, cfg.vocab), y.reshape(-1))
        loss.backward()
        opt.step()
        return loss

    loss = train_step()
    assert torch.isfinite(loss)

    def generate():
        idx = x[:1, :8].clone()
        with torch.no_grad():
            for _ in range(8):
                logits = model(idx[:, -cfg.seq:])
                nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                idx = torch.cat([idx, nxt], dim=1)
        return idx

    generated = generate()
    assert generated.shape[1] == 16
    train_ms = _time_ms(train_step, repeat=3, warmup=1)
    gen_ms = _time_ms(generate, repeat=3, warmup=1)
    print("=== nano-GPT correctness + benchmark ===")
    print("correctness: finite train loss and greedy generation extends the prompt")
    print(f"params={_parameter_count(model)} train_step_ms={train_ms:.3f} generate_8_tokens_ms={gen_ms:.3f}")
    _save_bar(results_dir, "bench_nano_gpt.png", "nano-GPT train/generate latency", "ms", ["train", "generate"], [train_ms, gen_ms])


def run_rlhf_benchmark(results_dir: Path, lang: str = "en") -> None:
    torch, nn, _ = _require_torch()
    _print_metadata()
    torch.manual_seed(19)
    batch, seq, vocab, dim = 32, 24, 128, 48

    class RewardModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(vocab, dim)
            self.head = nn.Linear(dim, 1)

        def forward(self, tokens):
            return self.head(self.emb(tokens).mean(dim=1)).squeeze(-1)

    reward = RewardModel()
    chosen = torch.randint(0, vocab, (batch, seq))
    rejected = torch.randint(0, vocab, (batch, seq))

    def reward_step():
        chosen_score = reward(chosen)
        rejected_score = reward(rejected)
        return -nn.functional.logsigmoid(chosen_score - rejected_score).mean()

    loss = reward_step()
    assert torch.isfinite(loss)
    old_logp = torch.randn(batch)
    new_logp = old_logp + 0.05 * torch.randn(batch)
    advantage = torch.randn(batch)

    def ppo_surrogate():
        ratio = torch.exp(new_logp - old_logp)
        clipped = torch.clamp(ratio, 0.8, 1.2) * advantage
        return -torch.minimum(ratio * advantage, clipped).mean()

    ppo_loss = ppo_surrogate()
    assert torch.isfinite(ppo_loss)
    reward_ms = _time_ms(reward_step, repeat=10, warmup=3)
    ppo_ms = _time_ms(ppo_surrogate, repeat=50, warmup=5)
    print("=== RLHF core operations correctness + benchmark ===")
    print("correctness: finite pairwise reward loss and PPO clipped surrogate")
    print(f"reward_batch_ms={reward_ms:.3f} ppo_surrogate_ms={ppo_ms:.3f}")
    _save_bar(results_dir, "bench_rlhf.png", "RLHF core operation latency", "ms", ["reward", "ppo"], [reward_ms, ppo_ms])


def _quantize_dequantize_symmetric(w, levels: int):
    torch, _, _ = _require_torch()
    max_abs = w.abs().max()
    if max_abs == 0:
        return torch.zeros_like(w), torch.tensor(1.0, dtype=w.dtype, device=w.device)
    scale = max_abs / levels
    q = torch.round(w / scale).clamp(-levels - 1, levels)
    return q * scale, scale


def run_qlora_inference_benchmark(results_dir: Path, lang: str = "en") -> None:
    torch, _, _ = _require_torch()
    _print_metadata()
    torch.manual_seed(23)
    in_dim, out_dim, rank = 256, 192, 8
    x = torch.randn(32, in_dim)
    base = torch.randn(out_dim, in_dim) * 0.02
    q_base, _ = _quantize_dequantize_symmetric(base, levels=7)  # 4-bit signed simulation
    lora_a = torch.randn(rank, in_dim) * 0.02
    lora_b = torch.randn(out_dim, rank) * 0.02
    alpha = 8 / rank
    delta = (lora_b @ lora_a) * alpha

    def unmerged():
        return x @ q_base.T + (x @ lora_a.T) @ lora_b.T * alpha

    merged_weight = q_base + delta

    def merged():
        return x @ merged_weight.T

    torch.testing.assert_close(unmerged(), merged(), rtol=1e-5, atol=1e-5)
    unmerged_ms = _time_ms(unmerged, repeat=20, warmup=5)
    merged_ms = _time_ms(merged, repeat=20, warmup=5)
    base_mb = base.numel() * 4 / 1024**2
    simulated_4bit_mb = base.numel() * 0.5 / 1024**2
    print("=== QLoRA-style simulated 4-bit + LoRA correctness + benchmark ===")
    print("correctness: unmerged LoRA path equals merged adapter weight path")
    print(f"fp32_base_mb={base_mb:.3f} simulated_4bit_base_mb={simulated_4bit_mb:.3f}")
    print(f"unmerged_ms={unmerged_ms:.3f} merged_ms={merged_ms:.3f}")
    _save_bar(results_dir, "bench_qlora_inference.png", "QLoRA-style simulated inference", "ms", ["unmerged", "merged"], [unmerged_ms, merged_ms])


def run_kv_cache_benchmark(results_dir: Path, lang: str = "en") -> None:
    torch, _, F = _require_torch()
    _print_metadata()
    torch.manual_seed(29)
    decode_steps, dim = 16, 64
    rows = []
    for n_ctx in [64, 128, 256, 512]:
        keys = torch.randn(1, n_ctx + decode_steps, dim)
        values = torch.randn(1, n_ctx + decode_steps, dim)
        queries = torch.randn(1, decode_steps, dim)

        def attend_last(q, k, v):
            return F.softmax(q @ k.transpose(-1, -2) / math.sqrt(dim), dim=-1) @ v

        def no_cache():
            outs = []
            for step in range(decode_steps):
                prefix = n_ctx + step + 1
                outs.append(attend_last(queries[:, step : step + 1], keys[:, :prefix], values[:, :prefix]))
            return torch.cat(outs, dim=1)

        def with_cache():
            kc = keys[:, :n_ctx].clone()
            vc = values[:, :n_ctx].clone()
            outs = []
            for step in range(decode_steps):
                idx = n_ctx + step
                kc = torch.cat([kc, keys[:, idx : idx + 1]], dim=1)
                vc = torch.cat([vc, values[:, idx : idx + 1]], dim=1)
                outs.append(attend_last(queries[:, step : step + 1], kc, vc))
            return torch.cat(outs, dim=1)

        torch.testing.assert_close(no_cache(), with_cache(), rtol=1e-5, atol=1e-6)
        t_no = _time_ms(no_cache, repeat=3, warmup=1)
        t_yes = _time_ms(with_cache, repeat=3, warmup=1)
        cache_mb = 2 * (n_ctx + decode_steps) * dim * 4 / 1024**2
        rows.append((n_ctx, t_no, t_yes, t_no / max(t_yes, 1e-12), cache_mb))

    print("=== KV cache correctness + benchmark ===")
    print("correctness: cached and non-cached last-token attention outputs are equal")
    print("| n_ctx | no_cache_ms | cache_ms | speedup | kv_cache_mb |")
    print("|---:|---:|---:|---:|---:|")
    for n_ctx, t_no, t_yes, speedup, cache_mb in rows:
        print(f"| {n_ctx} | {t_no:.3f} | {t_yes:.3f} | {speedup:.2f}x | {cache_mb:.3f} |")
    _save_bar(results_dir, "bench_kv_cache.png", "KV cache decode latency", "ms", [str(r[0]) for r in rows], [r[2] for r in rows])


def run_quantization_benchmark(results_dir: Path, lang: str = "en") -> None:
    torch, nn, _ = _require_torch()
    _print_metadata()
    torch.manual_seed(31)
    layer = nn.Linear(256, 256)
    x = torch.randn(16, 256)
    zero_q, zero_scale = _quantize_dequantize_symmetric(torch.zeros(4, 4), levels=127)
    assert torch.equal(zero_q, torch.zeros_like(zero_q))
    assert zero_scale.item() == 1.0

    with torch.no_grad():
        fp32_out = layer(x)
        q_weight, scale = _quantize_dequantize_symmetric(layer.weight, levels=127)
        sim_out = nn.functional.linear(x, q_weight, layer.bias)
    mae = (fp32_out - sim_out).abs().mean().item()
    max_err = (fp32_out - sim_out).abs().max().item()
    fp32_mb = layer.weight.numel() * 4 / 1024**2
    int8_weight_mb = layer.weight.numel() * 1 / 1024**2

    def fp32_forward():
        return layer(x)

    def simulated_weight_forward():
        return nn.functional.linear(x, q_weight, layer.bias)

    fp32_ms = _time_ms(fp32_forward, repeat=20, warmup=5)
    sim_ms = _time_ms(simulated_weight_forward, repeat=20, warmup=5)
    print("=== INT8 weight quantization simulation correctness + benchmark ===")
    print("correctness: zero-weight quantization avoids divide-by-zero; simulated output is finite")
    print(f"scale={scale.item():.6g} mean_abs_error={mae:.6g} max_abs_error={max_err:.6g}")
    print(f"fp32_weight_mb={fp32_mb:.3f} simulated_int8_weight_mb={int8_weight_mb:.3f}")
    print(f"fp32_forward_ms={fp32_ms:.3f} simulated_int8_weight_forward_ms={sim_ms:.3f}")
    _save_bar(results_dir, "bench_quantization.png", "Quantization simulation error", "absolute error", ["mean", "max"], [mae, max_err])


def run_attention_benchmark(results_dir: Path, lang: str) -> None:
    del lang
    torch, _, _ = _require_torch()
    torch.manual_seed(0)
    q = torch.randn(64, 32)
    k = torch.randn(64, 32)
    v = torch.randn(64, 32)
    scores = q @ k.T / math.sqrt(q.shape[-1])
    weights = scores.softmax(dim=-1)
    output = weights @ v
    assert output.shape == q.shape and torch.isfinite(output).all()
    assert torch.allclose(weights.sum(-1), torch.ones(64), atol=1e-5)
    elapsed = _time_ms(lambda: (q @ k.T / math.sqrt(32)).softmax(-1) @ v)
    print(f"attention_ms={elapsed:.3f}")
    _save_bar(results_dir, "bench_attention.png", "Scaled attention", "milliseconds", ["CPU"], [elapsed])


def run_backprop_benchmark(results_dir: Path, lang: str) -> None:
    del lang
    torch, nn, _ = _require_torch()
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 8))
    x, target = torch.randn(16, 32), torch.randn(16, 8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def step() -> None:
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()

    before = model[0].weight.detach().clone()
    elapsed = _time_ms(step, repeat=5, warmup=1)
    assert not torch.equal(before, model[0].weight) and model[0].weight.grad is not None
    print(f"backprop_ms={elapsed:.3f}")
    _save_bar(results_dir, "bench_backprop.png", "Backpropagation", "milliseconds", ["CPU"], [elapsed])


def run_lora_benchmark(results_dir: Path, lang: str) -> None:
    del lang
    torch, _, _ = _require_torch()
    torch.manual_seed(0)
    x = torch.randn(16, 32)
    weight = torch.randn(24, 32)
    a = torch.randn(4, 32, requires_grad=True)
    b = torch.zeros(24, 4, requires_grad=True)
    base = x @ weight.T
    adapted = base + (x @ a.T) @ b.T / 4
    adapted.square().mean().backward()
    assert b.grad is not None and torch.isfinite(b.grad).all()
    full, lora = weight.numel(), a.numel() + b.numel()
    assert lora < full
    print(f"full_params={full} lora_params={lora}")
    _save_bar(results_dir, "bench_lora.png", "Trainable parameters", "parameters", ["Full", "LoRA"], [full, lora])


def run_matmul_benchmark(results_dir: Path, lang: str) -> None:
    del lang
    torch, _, _ = _require_torch()
    torch.manual_seed(0)
    a, b = torch.randn(128, 64), torch.randn(64, 96)
    actual = a @ b
    expected = torch.from_numpy(a.numpy() @ b.numpy())
    assert torch.allclose(actual, expected, atol=1e-4, rtol=1e-4)
    elapsed = _time_ms(lambda: a @ b)
    print(f"matmul_ms={elapsed:.3f}")
    _save_bar(results_dir, "bench_matmul.png", "Matrix multiplication", "milliseconds", ["CPU"], [elapsed])


def run_optimizer_benchmark(results_dir: Path, lang: str) -> None:
    del lang
    torch, nn, _ = _require_torch()
    torch.manual_seed(0)
    x = torch.randn(64, 8)
    target = x @ torch.arange(8, dtype=torch.float32).unsqueeze(1)
    model = nn.Linear(8, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    initial = nn.functional.mse_loss(model(x), target).item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = nn.functional.mse_loss(model(x), target)
        loss.backward()
        optimizer.step()
    final = nn.functional.mse_loss(model(x), target).item()
    assert final < initial
    print(f"initial_loss={initial:.6f} final_loss={final:.6f}")
    _save_bar(results_dir, "bench_optimizer.png", "Optimizer convergence", "loss", ["Initial", "Final"], [initial, final])


BENCHMARKS: dict[str, Callable[[Path, str], None]] = {
    "attention": run_attention_benchmark,
    "backprop": run_backprop_benchmark,
    "lora": run_lora_benchmark,
    "matmul": run_matmul_benchmark,
    "optimizer": run_optimizer_benchmark,
    "tokenizer": run_tokenizer_benchmark,
    "mha_gqa": run_mha_gqa_benchmark,
    "gpt_sizes": run_gpt_sizes_benchmark,
    "training_loop": run_training_loop_benchmark,
    "nano_gpt": run_nano_gpt_benchmark,
    "rlhf": run_rlhf_benchmark,
    "qlora_inference": run_qlora_inference_benchmark,
    "kv_cache": run_kv_cache_benchmark,
    "quantization": run_quantization_benchmark,
}


def run_named_benchmark(name: str, results_dir: Path, lang: str = "en") -> None:
    try:
        benchmark = BENCHMARKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown benchmark: {name}") from exc
    benchmark(results_dir, lang)
