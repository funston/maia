import torch

from tinyserve import KVCache


def test_cached_decode_matches_uncached_forward(model, config, prompts):
    tokens = torch.tensor(prompts)
    batch, prompt_len = tokens.shape
    extra = torch.randint(1, config.vocab_size, (batch, 5))
    full = torch.cat([tokens, extra], dim=1)

    with torch.inference_mode():
        reference = model(full)

        cache = KVCache(config, batch)
        logits = model.prefill(tokens, torch.full((batch,), prompt_len), cache)
        torch.testing.assert_close(logits, reference[:, prompt_len - 1], atol=1e-4, rtol=1e-4)

        for i in range(extra.shape[1]):
            logits = model.decode_step(extra[:, i], cache)
            torch.testing.assert_close(
                logits, reference[:, prompt_len + i], atol=1e-4, rtol=1e-4
            )


def test_forward_is_causal(model, prompts):
    tokens = torch.tensor(prompts)
    changed = tokens.clone()
    changed[:, -1] = (changed[:, -1] + 1) % model.config.vocab_size
    with torch.inference_mode():
        a, b = model(tokens), model(changed)
    torch.testing.assert_close(a[:, :-1], b[:, :-1])
