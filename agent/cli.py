"""CLI entrypoint for the knowledge-base agent.

Designed for background operation:
    - Daemon mode: runs continuously, exploring and building KB
    - Single-run modes: explore, research, query
    - Status command: check agent progress
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
from agent.agent import Agent
from agent.prompts import EXPLORE_CODEBASE_PROMPT, RESEARCH_TOPIC_PROMPT, QUERY_KB_PROMPT


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
# Commands
# ═══════════════════════════════════════════════════════════════════════════


def cmd_explore(args: argparse.Namespace) -> None:
    """Explore a codebase and build KB entries."""
    agent = _make_agent(args)

    # Name the task
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

    # Name the task
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

    # Name the task
    task_dir = agent.name_task(f"Query: {query}")
    print(f"Task directory: {task_dir}/")

    write_status({"mode": "query", "state": "running", "query": query, "task_dir": task_dir})

    prompt = QUERY_KB_PROMPT.format(query=query, task_dir=task_dir)
    result = agent.run(prompt)
    print(result)

    write_status({"mode": "query", "state": "done", "task_dir": task_dir, **agent.stats()})


def cmd_daemon(args: argparse.Namespace) -> None:
    """Run continuously in the background, building knowledge."""
    if is_running():
        print(f"Agent already running (PID {is_running()}). Use 'stop' to stop it.")
        sys.exit(1)

    write_pid()
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger("agent.daemon")

    # Handle graceful shutdown
    shutdown = False

    def _handle_signal(signum, frame):
        nonlocal shutdown
        logger.info("Received signal %d, shutting down gracefully...", signum)
        shutdown = True

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    agent = _make_agent(args)
    interval = args.interval or 300  # Default 5 minutes between runs
    tasks = _build_daemon_tasks(args)

    logger.info("Daemon started (PID %d), interval=%ds, tasks=%d", os.getpid(), interval, len(tasks))
    write_status({"mode": "daemon", "state": "running", "pid": os.getpid(), "interval": interval})

    task_idx = 0
    while not shutdown:
        task_desc, task_prompt_tmpl = tasks[task_idx % len(tasks)]
        task_idx += 1

        # Name each task so it gets its own directory
        try:
            task_dir = agent.name_task(task_desc)
        except Exception as exc:
            logger.warning("Failed to name task, using fallback: %s", exc)
            task_dir = f"task-{task_idx}"
            agent.task_dir = task_dir

        task_prompt = task_prompt_tmpl.format(task_dir=task_dir, topic=task_desc)

        logger.info("Running task '%s' in %s/", task_desc[:100], task_dir)
        write_status({
            "mode": "daemon", "state": "running_task",
            "task": task_desc[:200], "task_dir": task_dir,
            "task_num": task_idx, "pid": os.getpid(),
        })

        try:
            result = agent.run(task_prompt)
            logger.info("Task complete. Result: %s", result[:200])
        except Exception as exc:
            logger.exception("Task failed: %s", exc)

        # Reset conversation between tasks to keep context fresh
        agent.reset_conversation()

        write_status({
            "mode": "daemon", "state": "sleeping",
            "next_run_at": time.time() + interval,
            "pid": os.getpid(),
            **agent.stats(),
        })

        # Sleep with shutdown check
        for _ in range(interval):
            if shutdown:
                break
            time.sleep(1)

    logger.info("Daemon stopped")
    write_status({"mode": "daemon", "state": "stopped", **agent.stats()})
    clear_pid()


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop a running daemon."""
    pid = is_running()
    if not pid:
        print("No agent is running.")
        return
    os.kill(pid, signal.SIGTERM)
    print(f"Sent SIGTERM to agent (PID {pid})")
    # Wait for it to exit
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

    # Show KB stats
    from agent.knowledge_base import KnowledgeBase
    kb = KnowledgeBase()
    stats = kb.stats()
    print(f"\nKnowledge base: {stats['entry_count']} entries, {stats['total_tokens']} tokens")
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


def _build_daemon_tasks(args: argparse.Namespace) -> list[tuple[str, str]]:
    """Build the rotation of tasks for daemon mode.

    Returns list of (description, prompt_template) tuples.
    Prompt templates may contain {task_dir} and {topic} placeholders.
    """
    tasks: list[tuple[str, str]] = []

    # Always explore codebase
    tasks.append(("Explore codebase", EXPLORE_CODEBASE_PROMPT))

    # User-specified topics
    topics = getattr(args, "topics", None) or []
    for topic in topics:
        tasks.append((topic, RESEARCH_TOPIC_PROMPT))

    if not tasks:
        tasks.append(("Explore codebase", EXPLORE_CODEBASE_PROMPT))

    return tasks


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
    p_daemon = sub.add_parser("daemon", help="Run continuously in background")
    p_daemon.add_argument("--codebase", type=str)
    p_daemon.add_argument("--kb-dir", type=str)
    p_daemon.add_argument("--interval", type=int, default=300, help="Seconds between tasks (default 300)")
    p_daemon.add_argument("--max-turns", type=int, default=30)
    p_daemon.add_argument("--topics", nargs="*", help="Topics to research in rotation")
    p_daemon.set_defaults(func=cmd_daemon)

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
