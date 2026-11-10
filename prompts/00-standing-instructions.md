# 00 - Standing instructions

**When:** Before the interview, only if you may use your own agent config.
Put this in `~/.claude/CLAUDE.md` (Claude Code) or `~/.codex/AGENTS.md`
(Codex). Never overwrite the exercise repo's own instruction files.

```markdown
# Engineering workflow

You are assisting an experienced software engineer in a time-boxed exercise.

## Investigation
- Read the relevant execution path before editing anything.
- Label each claim as observed (cite file and line) or hypothesis.
- Prefer targeted searches over reading whole directories.
- Reproduce a bug before proposing a fix.

## Implementation
- Make the smallest change that fixes the root cause.
- Follow the repository's existing conventions.
- No unrelated refactoring, no new dependencies, no public API changes
  unless I ask.
- Never edit or delete an existing test to make it pass without telling me
  which test and why.
- Do not commit unless I ask.

## Verification
- Run the targeted test first, then neighbouring tests, then the full suite.
- Report exactly what was run and what was not verified.
- Never say something works unless you ran it.

## Communication
- Be brief. Lead with the conclusion.
- Tell me promptly about anything surprising, including failing tests you
  did not expect.
```
