"""Core agent loop: LLM ↔ tool execution with retries and memory injection."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from agent import config
from agent.knowledge_base import KnowledgeBase
from agent.llm import (
    LLMError,
    LLMRetryExhausted,
    chat_completion,
    count_messages_tokens,
    extract_assistant_message,
    extract_tool_calls,
)
from agent.memory import LongTermMemory, SystemPrompt, WorkingMemory
from agent.prompts import SYSTEM_PROMPT, TASK_NAMING_PROMPT, TASK_NAMING_SYSTEM
from agent.tools import ToolRegistry, build_registry

logger = logging.getLogger(__name__)


class Agent:
    """Knowledge-base building agent with three-tier memory and tool use."""

    def __init__(
        self,
        *,
        codebase_root: str | Path | None = None,
        kb_dir: str | Path | None = None,
        system_prompt: str | None = None,
        max_turns: int | None = None,
    ):
        self.codebase_root = Path(codebase_root) if codebase_root else Path.cwd()

        # Memory tiers
        self.system_prompt = SystemPrompt(system_prompt or SYSTEM_PROMPT)
        self.ltm = LongTermMemory(memory_dir=kb_dir or config.KNOWLEDGE_BASE_DIR)
        self.wm = WorkingMemory()

        # Knowledge base
        self.kb = KnowledgeBase(base_dir=kb_dir or config.KNOWLEDGE_BASE_DIR)

        # Tools
        self.registry: ToolRegistry = build_registry(
            self.kb, self.ltm, self.wm, self.codebase_root
        )

        # Conversation history
        self.messages: list[dict[str, Any]] = []
        self.max_turns = max_turns or config.MAX_AGENT_TURNS

        # Task directory (set by name_task(), scopes KB writes)
        self.task_dir: str | None = None

        # Stats
        self.total_turns = 0
        self.total_tool_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    # ── Task naming ────────────────────────────────────────────────────

    def name_task(self, task_description: str) -> str:
        """Ask the LLM to generate a short directory slug for the task.

        Sets self.task_dir and creates the directory in the KB.
        Returns the slug.
        """
        prompt = TASK_NAMING_PROMPT.format(task_description=task_description)
        messages = [
            {"role": "system", "content": TASK_NAMING_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        response = chat_completion(messages, temperature=0.2, max_tokens=50)
        msg = extract_assistant_message(response)
        raw = msg.get("content", "").strip().strip("/").strip()

        # Sanitize to a valid directory slug
        slug = re.sub(r"[^a-z0-9-]", "-", raw.lower())
        slug = re.sub(r"-+", "-", slug).strip("-")
        slug = slug or "task"

        self.task_dir = slug
        # Ensure the directory exists
        task_path = Path(self.kb.base_dir) / slug
        task_path.mkdir(parents=True, exist_ok=True)
        logger.info("Task directory: %s", slug)
        return slug

    # ── Build the messages to send ─────────────────────────────────────

    def _build_messages(self) -> list[dict[str, Any]]:
        """Assemble the full message list with memory injection."""
        msgs: list[dict[str, Any]] = []

        # 1. System prompt
        msgs.append(self.system_prompt.to_message())

        # 2. Long-term memory context (injected as a system message)
        ltm_context = self.ltm.get_context_block()
        if ltm_context:
            msgs.append({
                "role": "system",
                "content": (
                    "# Long-Term Memory\n"
                    "The following is your persistent memory from prior sessions:\n\n"
                    f"{ltm_context}"
                ),
            })

        # 3. Working memory context
        wm_context = self.wm.get_context_block()
        if wm_context:
            msgs.append({
                "role": "system",
                "content": (
                    "# Working Memory (current session)\n\n"
                    f"{wm_context}"
                ),
            })

        # 4. Task directory context
        if self.task_dir:
            msgs.append({
                "role": "system",
                "content": (
                    "# Current Task Directory\n\n"
                    f"All knowledge base entries for this task MUST be written under "
                    f"the `{self.task_dir}/` directory prefix. For example, use "
                    f"`kb_write(\"{self.task_dir}/overview\", ...)` not `kb_write(\"overview\", ...)`.\n\n"
                    "You MUST also persist key findings and cross-session facts using "
                    "`memory_store` so they survive across sessions."
                ),
            })

        # 5. Conversation history
        msgs.extend(self.messages)

        return msgs

    # ── Single turn ────────────────────────────────────────────────────

    def _execute_turn(self) -> dict[str, Any]:
        """Execute one LLM call + tool execution cycle.

        Returns the assistant message dict.
        """
        messages = self._build_messages()
        tools = self.registry.get_schemas()

        # Track tokens for stats
        self.total_input_tokens += count_messages_tokens(messages)

        # LLM call (retries handled internally)
        response = chat_completion(messages, tools)

        # Extract usage stats
        usage = response.get("usage", {})
        self.total_output_tokens += usage.get("completion_tokens", 0)

        # Parse assistant message
        assistant_msg = extract_assistant_message(response)

        # Store in conversation history
        self.messages.append({
            "role": "assistant",
            "content": assistant_msg.get("content", ""),
            **({"tool_calls": assistant_msg["tool_calls"]} if "tool_calls" in assistant_msg else {}),
        })

        # Execute tool calls
        tool_calls = extract_tool_calls(assistant_msg)
        if tool_calls:
            for tc in tool_calls:
                self.total_tool_calls += 1
                name = tc["name"]
                args = tc["arguments"]
                logger.info("Executing tool: %s(%s)", name, _truncate_args(args))

                result = self.registry.execute(name, args)

                # Add tool result to conversation
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": name,
                    "content": result,
                })

        return assistant_msg

    # ── Main run loop ──────────────────────────────────────────────────

    def run(self, user_message: str) -> str:
        """Run the agent on a user message until completion.

        Returns the final assistant text response.
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_message})

        final_text = ""

        for turn in range(self.max_turns):
            self.total_turns += 1
            logger.info("=== Turn %d ===", self.total_turns)

            try:
                assistant_msg = self._execute_turn()
            except LLMRetryExhausted as exc:
                error_text = f"[Agent error: LLM call failed after {exc.attempts} retries: {exc.last_error}]"
                logger.error(error_text)
                self.messages.append({"role": "assistant", "content": error_text})
                return error_text
            except LLMError as exc:
                error_text = f"[Agent error: {exc}]"
                logger.error(error_text)
                self.messages.append({"role": "assistant", "content": error_text})
                return error_text

            # Check if the assistant wants to continue (has tool calls)
            tool_calls = extract_tool_calls(assistant_msg)
            text = assistant_msg.get("content", "") or ""

            if text:
                final_text = text

            # If no tool calls, the agent is done
            if not tool_calls:
                logger.info("Agent finished (no more tool calls)")
                break

            # Check finish reason
            # (some models signal stop even with tool calls)

        else:
            logger.warning("Agent hit max turns (%d)", self.max_turns)
            final_text += "\n\n[Reached maximum turns limit]"

        return final_text

    # ── Stream mode (yields intermediate results) ──────────────────────

    def run_streaming(self, user_message: str):
        """Generator that yields events as the agent runs.

        Yields dicts with type: "thinking", "tool_call", "tool_result", "text", "done", "error".
        """
        self.messages.append({"role": "user", "content": user_message})

        for turn in range(self.max_turns):
            self.total_turns += 1

            try:
                messages = self._build_messages()
                tools = self.registry.get_schemas()
                self.total_input_tokens += count_messages_tokens(messages)

                response = chat_completion(messages, tools)
                usage = response.get("usage", {})
                self.total_output_tokens += usage.get("completion_tokens", 0)

                assistant_msg = extract_assistant_message(response)

                self.messages.append({
                    "role": "assistant",
                    "content": assistant_msg.get("content", ""),
                    **({"tool_calls": assistant_msg["tool_calls"]} if "tool_calls" in assistant_msg else {}),
                })

                text = assistant_msg.get("content", "") or ""
                if text:
                    yield {"type": "text", "content": text, "turn": self.total_turns}

                tool_calls = extract_tool_calls(assistant_msg)
                if not tool_calls:
                    yield {"type": "done", "total_turns": self.total_turns, "total_tool_calls": self.total_tool_calls}
                    return

                for tc in tool_calls:
                    self.total_tool_calls += 1
                    name = tc["name"]
                    args = tc["arguments"]

                    yield {"type": "tool_call", "name": name, "arguments": args, "turn": self.total_turns}

                    result = self.registry.execute(name, args)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": name,
                        "content": result,
                    })

                    yield {"type": "tool_result", "name": name, "result": result[:500], "turn": self.total_turns}

            except LLMRetryExhausted as exc:
                yield {"type": "error", "message": f"LLM failed after {exc.attempts} retries: {exc.last_error}"}
                return
            except LLMError as exc:
                yield {"type": "error", "message": str(exc)}
                return

        yield {"type": "done", "total_turns": self.total_turns, "note": "max turns reached"}

    # ── Stats ──────────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        return {
            "total_turns": self.total_turns,
            "total_tool_calls": self.total_tool_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "ltm_tokens": self.ltm.total_tokens,
            "ltm_remaining": self.ltm.remaining_tokens,
            "wm_tokens": self.wm.total_tokens,
            "kb_stats": self.kb.stats(),
            "conversation_messages": len(self.messages),
        }

    def reset_conversation(self) -> None:
        """Clear conversation history and working memory (keeps LTM and KB)."""
        self.messages.clear()
        self.wm.clear()
        self.task_dir = None


def _truncate_args(args: dict, max_len: int = 100) -> str:
    s = json.dumps(args, default=str)
    if len(s) > max_len:
        return s[:max_len] + "..."
    return s
