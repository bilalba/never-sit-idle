# Never Sit Idle — Knowledge-Base Building Agent

## What this is
An autonomous agent harness that continuously gathers information from multiple sources and organizes it into a structured markdown knowledge base. Designed to run in the background as a daemon.

## Architecture

```
agent/
  config.py         — Config: API keys, token limits, retries (reads .env)
  llm.py            — OpenRouter client (Gemma 4 26B) with retries, token counting,
                      tool-call parsing (supports both OpenAI-style and Gemma's native format)
  memory.py         — Three-tier memory: SystemPrompt, LongTermMemory (50k tokens, persistent MD),
                      WorkingMemory (ephemeral per-session)
  knowledge_base.py — Nested MD file manager with index, CRUD, search, tagging
  sources.py        — Data source clients with centralized rate limiting:
                      Reddit (30/min), Wikipedia, HN, StackOverflow, GitHub, web fetcher
  tools.py          — 30 tools in a registry; LLM calls these to interact with everything
  prompts.py        — System prompts for explore/research/query modes
  agent.py          — Core loop: build messages → LLM call → parse tool calls → execute → repeat
  cli.py            — CLI with daemon mode, PID tracking, log files, graceful shutdown
tests/              — 137 tests covering all modules
```

## Key design decisions

- **Gemma tool-call parsing**: Gemma 4 uses `<|tool_call>call:name{...}<tool_call|>` with `<|"|>` string delimiters. The parser in `llm.py` normalizes this to standard JSON. It also handles when Gemma injects extra fields (like `id`) that aren't in the tool schema — `tools.py:execute()` filters arguments to declared params only.
- **Rate limiting**: All HTTP sources go through a centralized `RateLimiter` in `sources.py` that tracks both window-based (requests in last 60s) and header-based (`x-ratelimit-remaining`) limits. Reddit is capped at 30/min.
- **Memory budget**: LTM has a hard 50k token cap measured via tiktoken `cl100k_base`. Writes that would exceed the budget are rejected with ValueError.
- **Background-first**: The CLI daemon mode runs tasks in rotation with configurable intervals, logs to files, writes PID/status files, handles SIGTERM gracefully.
- **No React/Ink CLI**: Deliberately kept as plain Python since the agent primarily runs unattended in background.

## Running

```bash
# One-shot modes
python run.py explore --codebase /path/to/repo
python run.py research "topic"
python run.py query "question"

# Background daemon
python run.py daemon --topics "React hooks" "FastAPI" --interval 300
python run.py status
python run.py stop
python run.py tree
```

## Environment

- `.env` file at project root with `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`
- Python 3.13+, deps: requests, tiktoken, pytest

## Testing

```bash
python -m pytest tests/ -v
```

All 137 tests pass. Tests use mocks for HTTP/LLM calls — no live API calls in tests.

## Known issues / next steps

- Gemma sometimes produces malformed tool calls on the last turn (the content/args boundary bleeds). The parser handles most cases but very long content values can still break.
- No conversation compression yet — long research sessions may hit context limits.
- Could add: deduplication of KB entries, automatic summarization of long sources, embedding-based KB search.
