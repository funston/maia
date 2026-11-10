from __future__ import annotations

import torch

from tinyserve.config import ModelConfig


class KVCache:
    """Preallocated key/value storage for every layer of one batch.

    Buffers have shape ``[n_layers, batch, n_heads, capacity, head_dim]``.
    ``lengths[b]`` is the number of valid positions for sequence ``b``; slots
    at or beyond it hold padding or stale data and must never be attended to.

    A forward pass calls ``write`` once per layer and then ``advance`` once,
    so ``lengths`` is constant while the layers of one step are running.
    """

    def __init__(
        self,
        config: ModelConfig,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shape = (
            config.n_layers,
            batch_size,
            config.n_heads,
            config.max_seq_len,
            config.head_dim,
        )
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.lengths = torch.zeros(batch_size, dtype=torch.long, device=device)

    @property
    def batch_size(self) -> int:
        return self.k.shape[1]

    @property
    def capacity(self) -> int:
        return self.k.shape[3]

    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store ``T`` new positions for every sequence.

        ``k`` and ``v`` have shape ``[batch, n_heads, T, head_dim]``.
        """
        n_new = k.shape[2]
        start = int(self.lengths.max())
        if start + n_new > self.capacity:
            raise ValueError(
                f"KV cache overflow: {start} + {n_new} > capacity {self.capacity}"
            )
        self.k[layer, :, :, start : start + n_new] = k
        self.v[layer, :, :, start : start + n_new] = v

    def advance(self, n: int | torch.Tensor) -> None:
        """Mark ``n`` more positions as valid (scalar, or one count per sequence)."""
        self.lengths = self.lengths + n

    def view(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys and values for ``layer`` as ``[batch, n_heads, capacity, head_dim]``."""
        return self.k[layer], self.v[layer]

    def attention_mask(self, pending: int = 0) -> torch.Tensor:
        """Boolean mask ``[batch, 1, 1, capacity]``; True where attention is allowed.

        ``pending`` counts positions written during the current step that
        ``advance`` has not accounted for yet.
        """
        mask = torch.zeros(
            self.batch_size, self.capacity, dtype=torch.bool, device=self.k.device
        )
        for b in range(self.batch_size):
            mask[b, : int(self.lengths[b]) + pending] = True
        return mask[:, None, None, :]

    def select(self, index: torch.Tensor) -> None:
        """Keep only the sequences in ``index`` (used when sequences finish)."""
        self.k = self.k[:, index]
        self.v = self.v[:, index]
        self.lengths = self.lengths[: index.numel()]
