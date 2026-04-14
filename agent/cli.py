"""CLI entrypoint for the knowledge-base agent.

Designed for background operation:
    - Daemon mode: runs continuously, pulling jobs from a file-based queue
    - Single-run modes: explore, research, query
    - Queue management: add jobs, list jobs
    - Telegram bot: notifications + job intake when queue is empty
    - All output goes to log files
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

from agent import config
from agent import queue as Q
from agent.agent import Agent
from agent.memory import SystemPrompt
from agent.prompts import (
    EXPLORE_CODEBASE_PROMPT,
    RESEARCH_TOPIC_PROMPT,
    QUERY_KB_PROMPT,
    SYSTEM_PROMPT,
    TELEGRAM_CHAT_SYSTEM,
)
from agent.telegram import TelegramBot


LOG_DIR = Path(config.KNOWLEDGE_BASE_DIR) / ".logs"
PID_FILE = Path(config.KNOWLEDGE_BASE_DIR) / ".agent.pid"
STATUS_FILE = Path(config.KNOWLEDGE_BASE_DIR) / ".agent.status"


def setup_logging(verbose: bool = False, log_file: str | None = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = []

    # Always log to file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_path = log_file or str(LOG_DIR / "agent.log")
    fh = logging.FileHandler(file_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    handlers.append(fh)

    # Console only if not daemon
    if sys.stdout.isatty():
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S"))
        handlers.append(ch)

    logging.basicConfig(level=logging.DEBUG, handlers=handlers, force=True)


def write_status(status: dict) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps({**status, "updated_at": time.time()}, indent=2))


def write_pid() -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()))


def clear_pid() -> None:
    if PID_FILE.exists():
        PID_FILE.unlink()


def is_running() -> int | None:
    """Return PID if agent is running, else None."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # Check if process exists
        return pid
    except (ValueError, OSError):
        clear_pid()
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Job execution helpers
# ═══════════════════════════════════════════════════════════════════════════


def _prompt_for_job(job: dict, task_dir: str) -> str:
    """Build the LLM prompt for a given job."""
    job_type = job["type"]
    subject = job["subject"]

    if job_type == Q.EXPLORE:
        return EXPLORE_CODEBASE_PROMPT.format(task_dir=task_dir)
    elif job_type == Q.RESEARCH:
        return RESEARCH_TOPIC_PROMPT.format(topic=subject, task_dir=task_dir)
    elif job_type == Q.QUERY:
        return QUERY_KB_PROMPT.format(query=subject, task_dir=task_dir)
    else:
        return RESEARCH_TOPIC_PROMPT.format(topic=subject, task_dir=task_dir)


def _handle_telegram_message(
    agent: Agent,
    bot: TelegramBot,
    msg: 'IncomingMessage',
    logger: logging.Logger,
) -> None:
    """Route a plain-text Telegram message through the agent and reply."""
    # Ensure agent is in Telegram chat mode
    agent.system_prompt.text = TELEGRAM_CHAT_SYSTEM

    try:
        result = agent.run(msg.text)
    except Exception as exc:
        logger.exception("Agent error handling Telegram message: %s", exc)
        bot.reply(msg, f"Sorry, something went wrong: {exc}")
        return

    if result.strip():
        bot.send_long(result.strip(), reply_to=msg.message_id)
    else:
        bot.reply(msg, "(No response)")


def _run_job(agent: Agent, job: dict, logger: logging.Logger) -> None:
    """Execute a single job: name task, run agent, update job status."""
    job_id = job["id"]
    job_type = job["type"]
    subject = job["subject"]

    # Reset conversation context and restore the standard system prompt
    agent.reset_conversation()
    agent.system_prompt.text = SYSTEM_PROMPT

    Q.mark_running(job_id)

    # Name the task
    if job_type == Q.EXPLORE:
        task_desc = f"Explore codebase at {agent.codebase_root}"
    else:
        task_desc = f"{job_type.title()}: {subject}"

    try:
        task_dir = agent.name_task(task_desc)
    except Exception as exc:
        logger.warning("Failed to name task, using fallback: %s", exc)
        task_dir = f"task-{job_id}"
        agent.task_dir = task_dir

    # Set codebase root for explore jobs
    if job_type == Q.EXPLORE and subject:
        agent.codebase_root = Path(subject)

    prompt = _prompt_for_job(job, task_dir)

    logger.info("Running job %s: %s '%s' → %s/", job_id, job_type, subject[:80], task_dir)

    try:
        result = agent.run(prompt)
        logger.info("Job %s complete. Result: %s", job_id, result[:200])
        Q.mark_done(job_id, result_dir=task_dir)
    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id, exc)
        Q.mark_failed(job_id, error=str(exc))
        raise


# ═══════════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════════


def cmd_explore(args: argparse.Namespace) -> None:
    """Explore a codebase and build KB entries."""
    agent = _make_agent(args)

    task_desc = f"Explore codebase at {agent.codebase_root}"
    task_dir = agent.name_task(task_desc)
    print(f"Task directory: {task_dir}/")

    write_status({"mode": "explore", "state": "running", "codebase": str(agent.codebase_root), "task_dir": task_dir})

    prompt = EXPLORE_CODEBASE_PROMPT.format(task_dir=task_dir)
    print(f"Exploring codebase at {agent.codebase_root}...")
    for event in agent.run_streaming(prompt):
        _print_event(event)

    write_status({"mode": "explore", "state": "done", "task_dir": task_dir, **agent.stats()})
    print("\nDone. KB tree:")
    print(agent.kb.tree())


def cmd_research(args: argparse.Namespace) -> None:
    """Research a topic from multiple sources."""
    agent = _make_agent(args)
    topic = args.topic

    task_dir = agent.name_task(f"Research: {topic}")
    print(f"Task directory: {task_dir}/")

    write_status({"mode": "research", "state": "running", "topic": topic, "task_dir": task_dir})

    prompt = RESEARCH_TOPIC_PROMPT.format(topic=topic, task_dir=task_dir)
    print(f"Researching: {topic}")
    for event in agent.run_streaming(prompt):
        _print_event(event)

    write_status({"mode": "research", "state": "done", "task_dir": task_dir, **agent.stats()})


def cmd_query(args: argparse.Namespace) -> None:
    """Query the knowledge base."""
    agent = _make_agent(args)
    query = args.query

    task_dir = agent.name_task(f"Query: {query}")
    print(f"Task directory: {task_dir}/")

    write_status({"mode": "query", "state": "running", "query": query, "task_dir": task_dir})

    prompt = QUERY_KB_PROMPT.format(query=query, task_dir=task_dir)
    result = agent.run(prompt)
    print(result)

    write_status({"mode": "query", "state": "done", "task_dir": task_dir, **agent.stats()})


def cmd_daemon(args: argparse.Namespace) -> None:
    """Run continuously, pulling jobs from the queue.

    When the queue is empty:
      - If Telegram is configured, asks the user for work and waits for a reply
      - Otherwise, sleeps for --idle-timeout seconds then checks again
    """
    if is_running():
        print(f"Agent already running (PID {is_running()}). Use 'stop' to stop it.")
        sys.exit(1)

    write_pid()
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger("agent.daemon")

    # Graceful shutdown
    shutdown = False

    def _handle_signal(signum, frame):
        nonlocal shutdown
        logger.info("Received signal %d, shutting down gracefully...", signum)
        shutdown = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    agent = _make_agent(args)
    bot = TelegramBot()
    idle_timeout = args.idle_timeout
    jobs_completed = 0

    # Seed the queue with any --topics passed on the command line
    topics = getattr(args, "topics", None) or []
    for topic in topics:
        Q.add(Q.RESEARCH, topic)
        logger.info("Seeded queue with topic: %s", topic)

    logger.info(
        "Daemon started (PID %d), idle_timeout=%ds, telegram=%s, queued=%d",
        os.getpid(), idle_timeout, bot.enabled, Q.queue_size(),
    )
    if bot.enabled:
        bot.send("\u25b6\ufe0f *Agent started.* Watching queue for jobs.")

    write_status({"mode": "daemon", "state": "running", "pid": os.getpid()})

    while not shutdown:
        job = Q.next_job()

        if job:
            # ── Run the next job ──────────────────────────────────
            write_status({
                "mode": "daemon", "state": "running_job",
                "job_id": job["id"], "job_type": job["type"],
                "subject": job["subject"][:200],
                "pid": os.getpid(),
            })
            bot.notify_job_started(job)

            try:
                _run_job(agent, job, logger)
                jobs_completed += 1
                # Re-read the job to get updated fields
                done_job = Q.list_jobs()
                done_job = next((j for j in done_job if j["id"] == job["id"]), job)
                bot.notify_job_done(done_job)
            except Exception:
                failed_job = Q.list_jobs()
                failed_job = next((j for j in failed_job if j["id"] == job["id"]), job)
                bot.notify_job_failed(failed_job)

            # Drain any Telegram messages that arrived while the job ran
            if bot.enabled:
                pending = bot.poll_messages(timeout=0)
                if pending:
                    topics = bot.handle_commands(pending, Q, agent_stats=agent.stats())
                    for msg in topics:
                        logger.info("Routing Telegram message through agent (between jobs): %s", msg.text[:100])
                        _handle_telegram_message(agent, bot, msg, logger)

            write_status({
                "mode": "daemon", "state": "idle",
                "jobs_completed": jobs_completed,
                "jobs_queued": Q.queue_size(),
                "pid": os.getpid(),
                **agent.stats(),
            })
            continue  # immediately check for next job

        # ── Queue is empty ────────────────────────────────────────
        if bot.enabled:
            logger.info("Queue empty — waiting for Telegram messages")
            bot.notify_idle(completed_count=jobs_completed)

            write_status({
                "mode": "daemon", "state": "waiting_for_input",
                "jobs_completed": jobs_completed,
                "pid": os.getpid(),
            })

            # Long-poll Telegram — route messages through the agent
            while not shutdown:
                messages = bot.poll_messages(timeout=30)
                if not messages:
                    # Check if a job appeared (e.g. from external CLI `add`)
                    if Q.queue_size() > 0:
                        break
                    continue

                # Handle commands inline, get back plain-text messages
                topics = bot.handle_commands(messages, Q, agent_stats=agent.stats())

                # Route each message through the agent
                for msg in topics:
                    logger.info("Routing Telegram message through agent: %s", msg.text[:100])
                    _handle_telegram_message(agent, bot, msg, logger)

                # If the agent queued any jobs, break out to process them
                if Q.queue_size() > 0:
                    break

            if shutdown:
                break

            continue

        else:
            # No Telegram — just sleep and re-check
            logger.info("Queue empty, sleeping %ds...", idle_timeout)
            write_status({
                "mode": "daemon", "state": "idle_sleeping",
                "jobs_completed": jobs_completed,
                "jobs_queued": 0,
                "next_check_at": time.time() + idle_timeout,
                "pid": os.getpid(),
            })
            for _ in range(idle_timeout):
                if shutdown:
                    break
                # Wake up early if a job appears
                if Q.queue_size() > 0:
                    break
                time.sleep(1)

    # ── Shutdown ──────────────────────────────────────────────────────
    logger.info("Daemon stopped. Jobs completed: %d", jobs_completed)
    bot.notify_shutdown({"jobs_completed": jobs_completed, "jobs_queued": Q.queue_size()})
    write_status({"mode": "daemon", "state": "stopped", "jobs_completed": jobs_completed, **agent.stats()})
    clear_pid()


def cmd_add(args: argparse.Namespace) -> None:
    """Add a job to the queue."""
    job = Q.add(args.job_type, args.subject)
    print(f"Queued: [{job['type']}] {job['subject']}")
    print(f"Job ID: {job['id']}")
    pending = Q.queue_size()
    print(f"Queue: {pending} job{'s' if pending != 1 else ''} pending")


def cmd_jobs(args: argparse.Namespace) -> None:
    """List jobs in the queue."""
    status_filter = getattr(args, "filter", None)
    jobs = Q.list_jobs(status=status_filter)

    if not jobs:
        print("No jobs." if not status_filter else f"No {status_filter} jobs.")
        return

    for job in jobs:
        status = job["status"].upper()
        subject = job["subject"]
        if len(subject) > 60:
            subject = subject[:57] + "..."

        # Format timing
        timing = ""
        if job["status"] == Q.RUNNING and job.get("started_at"):
            elapsed = int(time.time() - job["started_at"])
            timing = f" ({elapsed}s ago)"
        elif job["status"] in (Q.DONE, Q.FAILED) and job.get("started_at") and job.get("completed_at"):
            duration = int(job["completed_at"] - job["started_at"])
            timing = f" ({duration}s)"

        result_dir = f" → {job['result_dir']}/" if job.get("result_dir") else ""
        error = f" !! {job['error'][:60]}" if job.get("error") else ""

        print(f"  [{status:7s}] {job['type']:8s}  {subject}{timing}{result_dir}{error}")


def cmd_clear(args: argparse.Namespace) -> None:
    """Clear completed/failed jobs from the queue."""
    removed = Q.clear_done()
    print(f"Removed {removed} completed/failed job{'s' if removed != 1 else ''}.")


def cmd_setup_telegram(args: argparse.Namespace) -> None:
    """Interactive setup for the Telegram bot integration."""
    env_path = Path(__file__).resolve().parent.parent / ".env"

    print("=== Telegram Bot Setup ===\n")
    print("Step 1: Create a bot")
    print("  1. Open Telegram and message @BotFather")
    print("  2. Send /newbot and follow the prompts")
    print("  3. Copy the bot token (looks like 123456:ABC-DEF...)\n")

    token = input("Paste your bot token: ").strip()
    if not token:
        print("No token provided, aborting.")
        return

    # Verify the token works
    import requests
    try:
        resp = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
        data = resp.json()
        if not data.get("ok"):
            print(f"Invalid token: {data.get('description', 'unknown error')}")
            return
        bot_name = data["result"].get("username", "unknown")
        print(f"  Bot verified: @{bot_name}\n")
    except requests.RequestException as exc:
        print(f"Could not verify token: {exc}")
        return

    print("Step 2: Get your chat ID")
    print(f"  1. Open Telegram and message @{bot_name}")
    print("  2. Send any message (e.g., 'hello')")
    input("  Press Enter once you've sent the message...")

    # Fetch updates to find the chat ID
    try:
        resp = requests.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=10)
        data = resp.json()
        updates = data.get("result", [])
    except requests.RequestException as exc:
        print(f"Could not fetch updates: {exc}")
        return

    chat_id = None
    for update in reversed(updates):  # most recent first
        msg = update.get("message", {})
        chat = msg.get("chat", {})
        if chat.get("id"):
            chat_id = str(chat["id"])
            chat_name = chat.get("first_name") or chat.get("title") or chat_id
            print(f"  Found chat: {chat_name} (ID: {chat_id})\n")
            break

    if not chat_id:
        print("  No messages found. Make sure you sent a message to the bot.")
        print("  You can set TELEGRAM_CHAT_ID manually in .env")
        chat_id = input("  Or enter your chat ID manually: ").strip()
        if not chat_id:
            print("No chat ID, aborting.")
            return

    # Send a test message
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": "\u2705 Bot connected! I'll notify you here when I need work."},
            timeout=10,
        )
        if resp.json().get("ok"):
            print("  Test message sent! Check your Telegram.\n")
        else:
            print(f"  Warning: test message failed: {resp.json().get('description')}\n")
    except requests.RequestException:
        pass

    # Write to .env
    env_lines = []
    if env_path.exists():
        env_lines = env_path.read_text().splitlines()

    # Remove existing telegram lines
    env_lines = [l for l in env_lines if not l.strip().startswith(("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"))]

    env_lines.append(f"TELEGRAM_BOT_TOKEN={token}")
    env_lines.append(f"TELEGRAM_CHAT_ID={chat_id}")

    env_path.write_text("\n".join(env_lines) + "\n")
    print(f"Saved to {env_path}")
    print("The daemon will now use Telegram for notifications and job intake.")


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop a running daemon."""
    pid = is_running()
    if not pid:
        print("No agent is running.")
        return
    os.kill(pid, signal.SIGTERM)
    print(f"Sent SIGTERM to agent (PID {pid})")
    for _ in range(10):
        if not is_running():
            print("Agent stopped.")
            return
        time.sleep(1)
    print("Agent still running. You may need to kill it manually.")


def cmd_status(args: argparse.Namespace) -> None:
    """Show agent status."""
    pid = is_running()
    print(f"Running: {'yes (PID ' + str(pid) + ')' if pid else 'no'}")

    if STATUS_FILE.exists():
        try:
            status = json.loads(STATUS_FILE.read_text())
            print(json.dumps(status, indent=2))
        except json.JSONDecodeError:
            print("(status file corrupt)")
    else:
        print("No status file found.")

    # Queue stats
    queued = Q.queue_size()
    total = len(Q.list_jobs())
    print(f"\nQueue: {queued} pending, {total} total")

    # KB stats
    from agent.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    stats = kb.stats()
    print(f"Knowledge base: {stats['entry_count']} entries, {stats['total_tokens']} tokens")
    if stats['categories']:
        print(f"Categories: {', '.join(stats['categories'])}")


def cmd_tree(args: argparse.Namespace) -> None:
    """Show KB tree."""
    from agent.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    print(kb.tree())


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_agent(args: argparse.Namespace) -> Agent:
    setup_logging(verbose=getattr(args, "verbose", False))
    return Agent(
        codebase_root=getattr(args, "codebase", None) or Path.cwd(),
        kb_dir=getattr(args, "kb_dir", None),
        max_turns=getattr(args, "max_turns", None),
    )


def _print_event(event: dict) -> None:
    etype = event.get("type", "")
    if etype == "text":
        print(f"\n{event['content']}")
    elif etype == "tool_call":
        args_str = json.dumps(event.get("arguments", {}), default=str)
        if len(args_str) > 120:
            args_str = args_str[:120] + "..."
        print(f"  -> {event['name']}({args_str})")
    elif etype == "tool_result":
        result = event.get("result", "")
        if len(result) > 200:
            result = result[:200] + "..."
        print(f"  <- {result}")
    elif etype == "error":
        print(f"  !! ERROR: {event['message']}")
    elif etype == "done":
        print(f"\n  [Done: {event.get('total_turns', '?')} turns, {event.get('total_tool_calls', '?')} tool calls]")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nsi-agent",
        description="Never Sit Idle — Knowledge-base building agent",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    # explore
    p_explore = sub.add_parser("explore", help="Explore a codebase")
    p_explore.add_argument("--codebase", type=str, help="Path to codebase root")
    p_explore.add_argument("--kb-dir", type=str, help="Knowledge base directory")
    p_explore.add_argument("--max-turns", type=int, default=30)
    p_explore.set_defaults(func=cmd_explore)

    # research
    p_research = sub.add_parser("research", help="Research a topic")
    p_research.add_argument("topic", type=str, help="Topic to research")
    p_research.add_argument("--kb-dir", type=str)
    p_research.add_argument("--max-turns", type=int, default=30)
    p_research.set_defaults(func=cmd_research)

    # query
    p_query = sub.add_parser("query", help="Query the knowledge base")
    p_query.add_argument("query", type=str, help="Question to answer")
    p_query.add_argument("--kb-dir", type=str)
    p_query.add_argument("--max-turns", type=int, default=20)
    p_query.set_defaults(func=cmd_query)

    # daemon
    p_daemon = sub.add_parser("daemon", help="Run continuously, processing queued jobs")
    p_daemon.add_argument("--codebase", type=str)
    p_daemon.add_argument("--kb-dir", type=str)
    p_daemon.add_argument("--idle-timeout", type=int, default=60,
                          help="Seconds to wait when queue is empty before re-checking (default 60)")
    p_daemon.add_argument("--max-turns", type=int, default=30)
    p_daemon.add_argument("--topics", nargs="*", help="Seed the queue with these research topics")
    p_daemon.set_defaults(func=cmd_daemon)

    # add (queue a job)
    p_add = sub.add_parser("add", help="Queue a job for the daemon")
    p_add.add_argument("job_type", choices=sorted(Q.VALID_TYPES),
                       help="Job type: explore, research, or query")
    p_add.add_argument("subject", type=str,
                       help="Topic to research, path to explore, or question to query")
    p_add.set_defaults(func=cmd_add)

    # jobs (list queue)
    p_jobs = sub.add_parser("jobs", help="List queued/running/completed jobs")
    p_jobs.add_argument("--filter", choices=[Q.QUEUED, Q.RUNNING, Q.DONE, Q.FAILED],
                        help="Filter by status")
    p_jobs.set_defaults(func=cmd_jobs)

    # clear (remove done/failed jobs)
    p_clear = sub.add_parser("clear", help="Remove completed/failed jobs from queue")
    p_clear.set_defaults(func=cmd_clear)

    # setup-telegram
    p_telegram = sub.add_parser("setup-telegram", help="Interactive Telegram bot setup")
    p_telegram.set_defaults(func=cmd_setup_telegram)

    # stop
    p_stop = sub.add_parser("stop", help="Stop running daemon")
    p_stop.set_defaults(func=cmd_stop)

    # status
    p_status = sub.add_parser("status", help="Show agent status")
    p_status.set_defaults(func=cmd_status)

    # tree
    p_tree = sub.add_parser("tree", help="Show knowledge base tree")
    p_tree.set_defaults(func=cmd_tree)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
