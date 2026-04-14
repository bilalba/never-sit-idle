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
  sources/          — Data source clients with centralized rate limiting:
    _base.py          RateLimiter, _rate_limited_get, HTML/RSS helpers
    _reddit.py        Reddit (30/min)
    _wikipedia.py     Wikipedia
    _hackernews.py    Hacker News (Firebase + Algolia)
    _stackexchange.py StackOverflow / StackExchange
    _github.py        GitHub (public, no auth)
    _web.py           Generic web page fetcher
    _news.py          Google News RSS, GDELT, Feedsearch
    _finance.py       Yahoo Finance, Alpha Vantage
  tools/            — 43 tools in a registry; LLM calls these to interact with everything
    _registry.py      ToolRegistry class and schema helpers
    _filesystem.py    File-system tool implementations (read, glob, grep)
    _definitions.py   All tool registrations wired to sources/kb/memory
  queue.py          — File-based job queue (JSON files in kb/.queue/jobs/)
  telegram.py       — Telegram bot: notifications, idle prompts, long-poll job intake
  prompts.py        — System prompts for explore/research/query modes
  agent.py          — Core loop: build messages → LLM call → parse tool calls → execute → repeat
  cli.py            — CLI with daemon mode, job queue, PID tracking, log files, graceful shutdown
tests/              — 197 tests covering all modules
```

## Key design decisions

- **Gemma tool-call parsing**: Gemma 4 uses `<|tool_call>call:name{...}<tool_call|>` with `<|"|>` string delimiters. The parser in `llm.py` normalizes this to standard JSON. It also handles when Gemma injects extra fields (like `id`) that aren't in the tool schema — `tools/_registry.py:execute()` filters arguments to declared params only.
- **Rate limiting**: All HTTP sources go through a centralized `RateLimiter` in `sources/_base.py` that tracks both window-based (requests in last 60s) and header-based (`x-ratelimit-remaining`) limits. Reddit is capped at 30/min.
- **Memory budget**: LTM has a hard 50k token cap measured via tiktoken `cl100k_base`. Writes that would exceed the budget are rejected with ValueError.
- **Job queue**: File-based queue in `kb/.queue/jobs/`. Each job is a JSON file with status transitions: queued → running → done/failed. The daemon polls for queued jobs; CLI `add` command writes new job files. No IPC — filesystem is the coordination layer.
- **Telegram bot**: When the queue drains, the daemon messages the user asking what to research next and long-polls for a reply. Also notifies on job start/done/fail/shutdown. Degrades gracefully — if no token configured, the daemon just sleeps and re-checks the queue.
- **Background-first**: The daemon runs in the foreground (no forking — Python antipattern). Use tmux/nohup/systemd to background it. Logs to files, writes PID/status files, handles SIGTERM gracefully.
- **No React/Ink CLI**: Deliberately kept as plain Python since the agent primarily runs unattended in background.

## Running

```bash
# One-shot modes
python run.py explore --codebase /path/to/repo
python run.py research "topic"
python run.py query "question"

# Queue jobs and start daemon
python run.py add research "React hooks"
python run.py add research "FastAPI best practices"
python run.py add explore /path/to/repo
python run.py daemon                              # processes queue, asks Telegram when idle
python run.py daemon --topics "React hooks"       # seeds queue then starts

# Queue management
python run.py jobs                                # list all jobs
python run.py jobs --filter queued                # just pending
python run.py clear                               # remove done/failed jobs
python run.py status
python run.py stop
python run.py tree
```

## Environment

- `.env` file at project root with `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `ALPHA_VANTAGE_API_KEY`
- Optional: `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` for Telegram notifications/job intake
- Python 3.13+, deps: requests, tiktoken, pytest

## Testing

```bash
python -m pytest tests/ -v
```

All 197 tests pass. Tests use mocks for HTTP/LLM calls — no live API calls in tests.

## Known issues / next steps

- Gemma sometimes produces malformed tool calls on the last turn (the content/args boundary bleeds). The parser handles most cases but very long content values can still break.
- No conversation compression yet — long research sessions may hit context limits.
- Could add: deduplication of KB entries, automatic summarization of long sources, embedding-based KB search.
