# 002: Decode throughput depends on max_seq_len, not on actual sequence length

**Reported by:** serving team
**Severity:** medium

We raised `max_seq_len` from 256 to 4096 to support a long-context workload.
Traffic did not change (prompts are still ~32 tokens, completions ~64), but
decode tokens/s fell sharply for everyone.

```
python bench/bench_decode.py --max-seq-len 256
python bench/bench_decode.py --max-seq-len 4096
```

Expected: for the same prompts and completion lengths, throughput should be
roughly independent of the configured maximum. Memory growing with
`max_seq_len` is understood and acceptable; time per token is not.
