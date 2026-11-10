# 12 - Course correction

**When:** The agent is exploring too long, widening the diff, or guessing.
Interrupt it; do not wait for it to finish.

**It is exploring without converging**

```text
Stop. Summarize in five lines what you have established so far and what is
still unknown. Then propose the single next step that most reduces the
uncertainty, and wait for me.
```

**The diff is growing beyond the fix**

```text
Stop. This change is larger than the bug requires. List each modified file
and say whether it is necessary for the fix. Revert everything that is not,
then re-run the regression test.
```

**It claims a cause without evidence**

```text
Show me the exact lines that prove that. If you cannot, treat it as a
hypothesis and give me the cheapest experiment that would confirm or refute
it.
```

**It changed a test to get green**

```text
You modified an existing test. Explain what it asserted before, why that
assertion was wrong or obsolete, and why changing it is not hiding a
regression. If you cannot justify it, restore the test.
```
