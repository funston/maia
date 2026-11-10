from __future__ import annotations

from typing import Sequence

import torch

from tinyserve.kv_cache import KVCache
from tinyserve.model import TinyTransformer


class Engine:
    """Greedy batched generation on top of ``TinyTransformer``."""

    def __init__(self, model: TinyTransformer, eos_token_id: int | None = None) -> None:
        self.model = model.eval()
        self.config = model.config
        self.eos_token_id = eos_token_id

    @torch.inference_mode()
    def generate(
        self, prompts: Sequence[Sequence[int]], max_new_tokens: int = 32
    ) -> list[list[int]]:
        """Return the generated tokens for each prompt, without the prompt or EOS."""
        if not prompts:
            return []
        if any(len(p) == 0 for p in prompts):
            raise ValueError("prompts must be non-empty")

        device = next(self.model.parameters()).device
        prompt_lens = torch.tensor([len(p) for p in prompts], device=device)
        longest = int(prompt_lens.max())
        if longest + max_new_tokens > self.config.max_seq_len:
            raise ValueError("prompt + max_new_tokens exceeds max_seq_len")

        tokens = torch.full(
            (len(prompts), longest), self.config.pad_token_id, device=device
        )
        for row, prompt in enumerate(prompts):
            tokens[row, : len(prompt)] = torch.tensor(prompt, device=device)

        outputs: list[list[int]] = [[] for _ in prompts]
        # active[i] is the index into ``prompts`` of the i-th row still decoding.
        active = torch.arange(len(prompts), device=device)

        cache = KVCache(self.config, len(prompts), device=device)
        logits = self.model.prefill(tokens, prompt_lens, cache)

        for step in range(max_new_tokens):
            next_tokens = logits.argmax(dim=-1)

            if self.eos_token_id is not None:
                keep = next_tokens != self.eos_token_id
                if not keep.all():
                    index = keep.nonzero(as_tuple=True)[0]
                    cache.select(index)
                    active = active[index]
                    next_tokens = next_tokens[index]
                    if active.numel() == 0:
                        break

            for row, token in zip(active.tolist(), next_tokens.tolist()):
                outputs[row].append(token)

            if step + 1 < max_new_tokens:
                logits = self.model.decode_step(next_tokens, cache)

        return outputs
