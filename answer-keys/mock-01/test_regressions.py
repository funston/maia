"""Regression tests the exercise should end up with (or equivalents).

Copy into interview-01-tinyserve/tests/ to grade a solution.
"""

import time

import torch

from tinyserve import Engine, KVCache, ModelConfig, TinyTransformer


def _ragged(prompts):
    return [prompts[0][:2], prompts[1], prompts[2][:4], prompts[3][:1]]


def test_ragged_batch_matches_single(model, prompts):
    """Defect 1: per-sequence write offset."""
    ragged = _ragged(prompts)
    engine = Engine(model)
    batched = engine.generate(ragged, max_new_tokens=16)
    single = [engine.generate([p], max_new_tokens=16)[0] for p in ragged]
    assert batched == single


def test_ragged_batch_with_early_eos_matches_single(model, prompts):
    """Defect 2: select() must reorder lengths along with k/v."""
    ragged = _ragged(prompts)
    free_run = [Engine(model).generate([p], max_new_tokens=16)[0] for p in ragged]
    # An EOS that stops the first (short) sequence early while others continue.
    eos = free_run[0][3]
    engine = Engine(model, eos_token_id=eos)
    batched = engine.generate(ragged, max_new_tokens=16)
    single = [engine.generate([p], max_new_tokens=16)[0] for p in ragged]
    assert len({len(o) for o in single}) > 1, "fixture should finish at different steps"
    assert batched == single


def test_select_reorders_lengths(config):
    cache = KVCache(config, batch_size=3)
    cache.advance(torch.tensor([2, 5, 7]))
    cache.select(torch.tensor([1, 2]))
    assert cache.lengths.tolist() == [5, 7]


def test_write_uses_per_sequence_offset(config):
    cache = KVCache(config, batch_size=2)
    cache.advance(torch.tensor([1, 4]))
    one = torch.ones(2, config.n_heads, 1, config.head_dim)
    cache.write(0, one, one)
    keys = cache.k[0]  # raw buffer, independent of how view() slices
    assert torch.all(keys[0, :, 1] == 1.0)
    assert torch.all(keys[1, :, 4] == 1.0)
    assert torch.all(keys[0, :, 4] == 0.0)


def test_decode_time_independent_of_capacity():
    """Defect 3: coarse guard; a 16x larger max_seq_len must not cost 2x time."""

    def best_time(max_seq_len):
        torch.manual_seed(0)
        engine = Engine(TinyTransformer(ModelConfig(max_seq_len=max_seq_len)))
        prompts = torch.randint(1, 256, (8, 32)).tolist()
        engine.generate(prompts, max_new_tokens=64)
        best = float("inf")
        for _ in range(5):
            start = time.perf_counter()
            engine.generate(prompts, max_new_tokens=64)
            best = min(best, time.perf_counter() - start)
        return best

    assert best_time(4096) < 2.0 * best_time(256)
