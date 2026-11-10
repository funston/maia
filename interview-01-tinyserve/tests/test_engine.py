import pytest

from tinyserve import Engine


def test_respects_max_new_tokens(model, prompts):
    out = Engine(model).generate(prompts, max_new_tokens=10)
    assert [len(o) for o in out] == [10] * len(prompts)


def test_batch_matches_single(model, prompts):
    engine = Engine(model)
    batched = engine.generate(prompts, max_new_tokens=12)
    single = [engine.generate([p], max_new_tokens=12)[0] for p in prompts]
    assert batched == single


def test_stops_at_eos(model, prompts):
    free_run = Engine(model).generate(prompts, max_new_tokens=12)
    eos = free_run[0][4]

    out = Engine(model, eos_token_id=eos).generate(prompts, max_new_tokens=12)

    for free, stopped in zip(free_run, out):
        expected = free[: free.index(eos)] if eos in free else free
        assert stopped == expected


def test_handles_prompts_of_different_lengths(model, prompts):
    ragged = [prompts[0][:2], prompts[1], prompts[2][:4]]
    out = Engine(model).generate(ragged, max_new_tokens=8)
    assert [len(o) for o in out] == [8, 8, 8]
    assert all(0 <= t < model.config.vocab_size for o in out for t in o)


def test_empty_batch(model):
    assert Engine(model).generate([]) == []


def test_rejects_empty_prompt(model):
    with pytest.raises(ValueError):
        Engine(model).generate([[]])


def test_rejects_too_long(model, config):
    with pytest.raises(ValueError):
        Engine(model).generate([[1, 2, 3]], max_new_tokens=config.max_seq_len)
