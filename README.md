# AI-assisted coding interview practice

Practice material for interviews where you are handed an unfamiliar
repository, an issue, and an AI coding agent.

| Path | What |
| --- | --- |
| [`prompts/`](prompts/) | Labeled, copy-paste prompts for each phase of a session |
| [`interview-01-tinyserve/`](interview-01-tinyserve/) | Mock exercise: a small PyTorch inference engine with open issues |
| [`answer-keys/`](answer-keys/) | **Spoilers.** Root causes, hidden regression tests, reference fix |

## Running a mock session

1. Set a 60-minute timer.
2. Start your agent **inside** `interview-01-tinyserve/` so the answer key is
   outside its working directory. For a stricter run, copy that folder
   somewhere else first:
   ```
   cp -R interview-01-tinyserve /tmp/tinyserve && cd /tmp/tinyserve && git init -q && git add -A && git commit -qm "start"
   ```
3. Read `TASK.md`, then work through the prompts in `prompts/` in order.
4. Narrate your decisions out loud as you would to an interviewer.
5. Afterwards, grade yourself with `answer-keys/mock-01/ANSWER-KEY.md`.

## The loop

Discover -> Investigate -> Reproduce -> Fix -> Review -> Explain.

You own every transition. The agent proposes; you decide what is true.
