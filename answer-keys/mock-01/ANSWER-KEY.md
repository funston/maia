# Answer key: interview-01-tinyserve

**Spoilers.** Do not open this before attempting the exercise, and do not
start your agent in a directory where it can read this folder.

## Grading a solution

```
cp answer-keys/mock-01/test_regressions.py interview-01-tinyserve/tests/
cd interview-01-tinyserve && pytest -q
```

Unfixed repo: 13 pass, 5 regressions fail. `fix.patch` is a reference
solution (`git apply answer-keys/mock-01/fix.patch` from the repo root); with
it all 18 pass and the benchmark is flat across `max_seq_len`.

## Issue 001 has two defects, not one

### Defect 1: write offset is the batch maximum (`KVCache.write`)

`start = int(self.lengths.max())` writes every sequence's new key/value at
the longest sequence's position. For a shorter sequence `b`:

- the new token's K/V lands in slot `max_len`, which the mask hides;
- slot `lengths[b]`, which the mask exposes, still holds the K/V of a **pad
  token** written during prefill.

So the shorter sequences attend to padding instead of their own latest token.
This explains every clue in the issue: only some prompts are affected (the
longest in the batch is always right), the first token agrees (it comes from
prefill, which does not read the cache), and fixed-size prompts never show it.

Fix: write at `lengths[b] + arange(T)` per sequence.

### Defect 2: `KVCache.select` does not reorder `lengths`

`self.lengths = self.lengths[: index.numel()]` truncates instead of indexing.
When a sequence finishes and the batch is compacted, the survivors inherit the
lengths of the first rows. Invisible when all lengths are equal, which is why
`test_stops_at_eos` and `test_select_keeps_requested_rows` pass. This is the
"it got worse with a real EOS" clue.

Fix: `self.lengths = self.lengths[index]`.

**The trap:** after fixing defect 1, a ragged-batch regression test goes
green and the agent declares victory. The issue text says EOS made it worse;
a candidate who re-reads the issue against the fix, or asks for a skeptical
review of the diff, finds defect 2. A ragged batch *with* early EOS is the
test that proves the whole issue is closed.

## Issue 002: attention runs over the full capacity

`KVCache.view` returns all `max_seq_len` slots and `attention_mask` builds a
`[batch, capacity]` mask with a Python loop over the batch, every layer, every
step. Cost per token scales with configured capacity, not sequence length.

Fix: slice keys, values and mask to `lengths.max() + pending` (slices are
views, so no copy) and build the mask with one vectorized comparison.

Wrinkle: `test_attention_mask_covers_valid_positions` asserts the mask is
`max_seq_len` wide, so it encodes the slow behavior and must be updated. A
good candidate says so explicitly rather than letting the agent quietly edit
a test to make it pass.

Measured on the reference machine (CPU, batch 8, 32-token prompts, 64 new):

| max_seq_len | before | after |
| --- | --- | --- |
| 256 | ~23.7k tok/s | ~24.7k tok/s |
| 4096 | ~8.9k tok/s | ~24.1k tok/s |

## Weak tests planted on purpose

- `test_batch_matches_single`: equal-length prompts only.
- `test_handles_prompts_of_different_lengths`: checks counts and vocab range,
  proves nothing about correctness.
- `test_cached_decode_matches_uncached_forward`: equal lengths only.

## What a strong session looks like

1. Reads the issue clues and predicts "ragged lengths + cache" before
   opening much code.
2. Gets a failing ragged-batch test first, and confirms it fails for the
   expected reason.
3. States the root cause in terms of which slot is written and which is read.
4. Does not accept wrong-direction fixes: left-padding, zeroing pad K/V,
   or masking pads in prefill all hide defect 1 without fixing the offset.
5. Adds the EOS + ragged test and finds defect 2.
6. For 002, measures before and after, and notices the test that pins the
   slow behavior.
7. Final diff touches `kv_cache.py`, one line of `model.py`, and tests.

## Interviewer-style follow-ups to rehearse

- Why does the first generated token always match?
- Why is the longest prompt in a batch never affected?
- `select` uses advanced indexing, which copies the whole cache each time a
  sequence finishes. When would that matter and what would you do instead?
- What changes if prompts are left-padded?
- How would this design need to change for continuous batching (new
  requests joining mid-decode)?
- The perf test compares wall-clock times. Is that acceptable in CI?
