"""Decode throughput benchmark.

Reports generated tokens per second for a fixed synthetic workload.
"""

from __future__ import annotations

import argparse
import time

import torch

from tinyserve import Engine, ModelConfig, TinyTransformer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(0)
    config = ModelConfig(max_seq_len=args.max_seq_len)
    engine = Engine(TinyTransformer(config))
    prompts = torch.randint(
        1, config.vocab_size, (args.batch_size, args.prompt_len)
    ).tolist()

    engine.generate(prompts, max_new_tokens=args.new_tokens)  # warm-up

    best = float("inf")
    for _ in range(args.repeats):
        start = time.perf_counter()
        engine.generate(prompts, max_new_tokens=args.new_tokens)
        best = min(best, time.perf_counter() - start)

    generated = args.batch_size * args.new_tokens
    print(
        f"max_seq_len={args.max_seq_len} batch={args.batch_size} "
        f"prompt={args.prompt_len} new={args.new_tokens}: "
        f"{generated / best:,.0f} tok/s (best of {args.repeats})"
    )


if __name__ == "__main__":
    main()
