from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tinyserve.config import ModelConfig
from tinyserve.kv_cache import KVCache


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, dim = x.shape
        return x.view(batch, seq, self.n_heads, dim // self.n_heads).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        layer: int,
        cache: KVCache | None = None,
        prefill: bool = True,
    ) -> torch.Tensor:
        batch, seq, dim = x.shape
        q, k, v = (self._split_heads(t) for t in self.qkv(x).chunk(3, dim=-1))

        if cache is not None:
            cache.write(layer, k, v)

        if cache is None or prefill:
            # Prompts are right-padded, so the causal mask alone keeps real
            # tokens from seeing padding.
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            keys, values = cache.view(layer)
            mask = cache.attention_mask(pending=seq)
            out = F.scaled_dot_product_attention(q, keys, values, attn_mask=mask)

        out = out.transpose(1, 2).reshape(batch, seq, dim)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = Attention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        layer: int,
        cache: KVCache | None = None,
        prefill: bool = True,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), layer, cache, prefill)
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        # Untrained weights; a wide init keeps greedy decoding from collapsing
        # onto a single token, which would make the tests meaningless.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.3)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def _embed(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.tok_emb(tokens) + self.pos_emb(positions)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Uncached reference path. ``tokens``: ``[batch, T]`` -> logits ``[batch, T, vocab]``."""
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self._embed(tokens, positions)
        for i, block in enumerate(self.blocks):
            x = block(x, i)
        return self.head(self.ln_f(x))

    def prefill(
        self, tokens: torch.Tensor, prompt_lens: torch.Tensor, cache: KVCache
    ) -> torch.Tensor:
        """Run right-padded prompts and fill ``cache``.

        Returns logits ``[batch, vocab]`` at each sequence's last real token.
        """
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self._embed(tokens, positions)
        for i, block in enumerate(self.blocks):
            x = block(x, i, cache, prefill=True)
        cache.advance(prompt_lens)
        rows = torch.arange(tokens.shape[0], device=tokens.device)
        last = self.ln_f(x)[rows, prompt_lens - 1]
        return self.head(last)

    def decode_step(self, tokens: torch.Tensor, cache: KVCache) -> torch.Tensor:
        """Feed one token per sequence. ``tokens``: ``[batch]`` -> logits ``[batch, vocab]``."""
        x = self._embed(tokens, cache.lengths)[:, None, :]
        for i, block in enumerate(self.blocks):
            x = block(x, i, cache, prefill=False)
        cache.advance(1)
        return self.head(self.ln_f(x[:, 0]))
