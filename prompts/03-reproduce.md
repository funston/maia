# 03 - Reproduce with a failing test

**When:** You have a leading hypothesis. Budget: 8 minutes.

```text
Write the smallest regression test that demonstrates the reported bug.

- Follow the existing test conventions and fixtures.
- Exercise the real code path; do not mock the behaviour under test.
- Cover the boundary case the issue describes.
- Run only this test and show me the failure output.
- Confirm it fails for the reason we expect, not for an unrelated one.

Do not change the implementation yet. If the bug cannot be reproduced
reliably, stop and tell me why, and propose the next best way to verify it.
```
