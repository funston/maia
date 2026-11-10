# 05 - Skeptical review

**When:** Tests are green. Do this even if you are confident. Budget: 10 minutes.

```text
Review the current git diff as a skeptical senior engineer who assumes it
contains a subtle bug. Do not modify files.

Check specifically:
1. Re-read the original issue. Does the fix explain and resolve every
   symptom it lists, or only the one we tested?
2. Boundary conditions and inputs our new test does not cover.
3. Other callers or code paths with the same flaw.
4. Regressions in adjacent behaviour.
5. Tests that pass without proving the intended behaviour, including
   existing ones.
6. Anything in the diff that is not needed for the fix.

List concrete findings by severity with file and line. If you find nothing
significant, say so; do not invent problems.

[ORIGINAL ISSUE TEXT]
```
