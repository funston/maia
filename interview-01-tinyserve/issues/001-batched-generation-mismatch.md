# 001: Batched generation returns different tokens than one-at-a-time

**Reported by:** eval team
**Severity:** high (silently wrong output)

Our eval harness sends prompts through `Engine.generate` in batches of 8. The
scores moved when we changed the batch size, which should not be possible with
greedy decoding. Running the same prompts one at a time gives different
completions for some of them.

What we know:

- Not every prompt is affected. In a given batch some completions are
  identical to the single-prompt run and some are not.
- The first generated token always seems to agree. Divergence starts later.
- It got noticeably worse once we turned on a real EOS token for the eval
  set, but we also see it with EOS disabled.
- We could not reproduce it with the synthetic load generator, which sends
  fixed-size prompts.

Expected: for greedy decoding, `engine.generate([a, b, c])` returns exactly
`[engine.generate([a])[0], engine.generate([b])[0], engine.generate([c])[0]]`.
