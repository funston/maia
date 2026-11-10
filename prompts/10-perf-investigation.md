# 10 - Performance investigation

**When:** The issue is a slowdown or scaling problem rather than wrong output.

```text
Investigate this performance issue. Do not modify code yet.

[PASTE ISSUE TEXT]

1. Find or write the smallest benchmark that shows the problem, run it, and
   record the baseline numbers.
2. Identify which input dimension the cost scales with and which it should
   scale with.
3. Walk the hot path and list every operation whose cost depends on the
   wrong dimension: full-buffer copies, allocations inside loops, Python
   loops over tensors, masks or tensors built at maximum size, device or
   dtype conversions, host/device synchronization.
4. Rank the candidates by expected impact and say how you would confirm
   each one (profiler, timing, or reasoning from shapes).

Separate what you measured from what you infer. After I pick a candidate,
the fix must keep outputs identical and be re-measured with the same
benchmark.
```
