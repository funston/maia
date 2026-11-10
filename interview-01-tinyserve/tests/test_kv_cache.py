import pytest
import torch

from tinyserve import KVCache, ModelConfig


def _kv(config: ModelConfig, batch: int, seq: int, fill: float) -> torch.Tensor:
    return torch.full((batch, config.n_heads, seq, config.head_dim), fill)


def test_write_then_advance_tracks_lengths(config):
    cache = KVCache(config, batch_size=2)
    cache.write(0, _kv(config, 2, 5, 1.0), _kv(config, 2, 5, 2.0))
    cache.advance(5)
    assert cache.lengths.tolist() == [5, 5]

    cache.write(0, _kv(config, 2, 1, 3.0), _kv(config, 2, 1, 4.0))
    cache.advance(1)
    assert cache.lengths.tolist() == [6, 6]

    keys, values = cache.view(0)
    assert torch.all(keys[:, :, :5] == 1.0)
    assert torch.all(keys[:, :, 5] == 3.0)
    assert torch.all(values[:, :, 5] == 4.0)


def test_attention_mask_covers_valid_positions(config):
    cache = KVCache(config, batch_size=2)
    cache.write(0, _kv(config, 2, 3, 1.0), _kv(config, 2, 3, 1.0))
    cache.advance(3)
    mask = cache.attention_mask(pending=1)
    assert mask.shape == (2, 1, 1, config.max_seq_len)
    assert mask[0, 0, 0].sum().item() == 4


def test_overflow_raises(config):
    cache = KVCache(config, batch_size=1)
    too_long = config.max_seq_len + 1
    with pytest.raises(ValueError, match="overflow"):
        cache.write(0, _kv(config, 1, too_long, 0.0), _kv(config, 1, too_long, 0.0))


def test_select_keeps_requested_rows(config):
    cache = KVCache(config, batch_size=3)
    k = torch.arange(3.0).view(3, 1, 1, 1).expand(3, config.n_heads, 2, config.head_dim)
    cache.write(0, k, k)
    cache.advance(2)

    cache.select(torch.tensor([0, 2]))

    assert cache.batch_size == 2
    assert cache.lengths.tolist() == [2, 2]
    keys, _ = cache.view(0)
    assert torch.all(keys[0, :, :2] == 0.0)
    assert torch.all(keys[1, :, :2] == 2.0)
