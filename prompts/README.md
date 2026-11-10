# Prompts

One prompt per file, numbered in the order you would normally use them.
Each file has a one-line **When**, then the prompt in a code block to paste
as-is. Replace anything in `[BRACKETS]`.

## Core loop (bug fix in an unfamiliar repo)

| # | File | Use it to |
| --- | --- | --- |
| 01 | [01-recon.md](01-recon.md) | Map the repo and find the build/test commands |
| 02 | [02-investigate.md](02-investigate.md) | Trace the issue to a root cause, no edits |
| 03 | [03-reproduce.md](03-reproduce.md) | Get a failing regression test |
| 04 | [04-fix.md](04-fix.md) | Make the smallest correct change |
| 05 | [05-skeptical-review.md](05-skeptical-review.md) | Attack the diff before you present it |
| 06 | [06-wrap-up.md](06-wrap-up.md) | Produce the summary you will talk through |

## Variants

| # | File | Use it when |
| --- | --- | --- |
| 10 | [10-perf-investigation.md](10-perf-investigation.md) | The issue is "it's slow", not "it's wrong" |
| 11 | [11-feature-plan.md](11-feature-plan.md) | The task is a feature, not a bug |
| 12 | [12-course-correct.md](12-course-correct.md) | The agent is drifting, sprawling, or guessing |
| 13 | [13-explain-code.md](13-explain-code.md) | You need to understand one piece well enough to defend it |

## Setup

| File | What |
| --- | --- |
| [00-standing-instructions.md](00-standing-instructions.md) | Contents for a user-level `CLAUDE.md` / `AGENTS.md`, if the interview allows your own config |

## Time budget for 60 minutes

| Minutes | Phase | Prompt |
| --- | --- | --- |
| 0-5 | Discover | 01 |
| 5-12 | Localize | 02 |
| 12-20 | Reproduce | 03 |
| 20-42 | Implement | 04 |
| 42-53 | Review | 05 |
| 53-60 | Explain | 06 |

Merge 01 and 02 for a small repo. Never skip 03 or 05.
