# 04 - Minimal fix

**When:** The regression test fails for the right reason. Budget: 20 minutes.

```text
Implement the smallest correct fix for the reproduced issue.

- Fix the root cause, not the symptom the test happens to check.
- Keep public APIs and unrelated behaviour unchanged.
- Follow the existing style. No unrelated refactoring, no new dependencies.
- Keep the regression test. Do not weaken or delete existing tests; if one
  must change, stop and tell me which and why.
- Run the regression test, then the neighbouring tests, then the full suite.
- If something fails, diagnose it before changing more code.

When done, give me: what changed and why, the test results you actually
ran, and anything still unverified. Do not commit.
```
