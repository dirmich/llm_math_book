#!/usr/bin/env python3
"""Run each shared benchmark implementation once with correctness assertions."""

from __future__ import annotations

import argparse
import platform
import tempfile
from pathlib import Path

import torch

from llm_math.experiments.benchmarks import run_named_benchmark


BENCHMARKS = (
    "attention",
    "backprop",
    "lora",
    "matmul",
    "optimizer",
    "tokenizer",
    "mha_gqa",
    "gpt_sizes",
    "training_loop",
    "nano_gpt",
    "rlhf",
    "qlora_inference",
    "kv_cache",
    "quantization",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--lang", default="ko,en,jp")
    parser.add_argument("--record-environment", action="store_true")
    parser.add_argument("--assert-correctness", action="store_true")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.record_environment:
        print(f"python={platform.python_version()} torch={torch.__version__} platform={platform.platform()}")
    if args.device == "cuda":
        torch.manual_seed(0)
        left = torch.randn(128, 64)
        right = torch.randn(64, 96)
        expected = left @ right
        torch.cuda.manual_seed_all(0)
        device = torch.device("cuda")
        left_cuda = left.to(device)
        right_cuda = right.to(device)
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(3):
            _ = left_cuda @ right_cuda
        torch.cuda.synchronize(device)
        actual = left_cuda @ right_cuda
        torch.cuda.synchronize()
        actual_cpu = actual.cpu()
        max_abs_error = (actual_cpu - expected).abs().max().item()
        max_rel_error = ((actual_cpu - expected).abs() / expected.abs().clamp_min(1e-12)).max().item()
        if not torch.allclose(actual_cpu, expected, atol=1e-3, rtol=1e-3):
            raise SystemExit("CUDA matrix multiplication failed the CPU reference check")
        cuda_version = getattr(getattr(torch, "version", None), "cuda", "unknown")
        capability = ".".join(str(part) for part in torch.cuda.get_device_capability(device))
        print(
            f"cuda={cuda_version} device={torch.cuda.get_device_name(device)} "
            f"compute_capability={capability} correctness=pass seed=0 warmup=3 "
            f"atol=1e-3 rtol=1e-3 max_abs_error={max_abs_error:.9g} "
            f"max_rel_error={max_rel_error:.9g} "
            f"memory_allocated_bytes={torch.cuda.memory_allocated(device)} "
            f"memory_reserved_bytes={torch.cuda.memory_reserved(device)} "
            f"max_memory_allocated_bytes={torch.cuda.max_memory_allocated(device)}"
        )
        return 0
    languages = args.lang.split(",")
    if any(language not in {"ko", "en", "jp"} for language in languages):
        raise SystemExit("--lang must contain only ko,en,jp")
    with tempfile.TemporaryDirectory(prefix="llm-math-benchmarks-") as output_dir:
        results = Path(output_dir)
        for language in languages:
            for name in BENCHMARKS:
                print(f"RUN {language}/{name}")
                run_named_benchmark(name, results / language, lang=language)
    print(f"OK: {len(BENCHMARKS) * len(languages)} language benchmark paths passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
