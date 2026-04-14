"""File-based job queue.

Jobs are JSON files in QUEUE_DIR/jobs/. The daemon polls this directory
for queued jobs, runs them oldest-first, and marks them done/failed.
CLI commands create new job files directly — no IPC needed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from agent import config

QUEUE_DIR = Path(config.KNOWLEDGE_BASE_DIR) / ".queue"
JOBS_DIR = QUEUE_DIR / "jobs"

# Valid states
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

# Valid job types
RESEARCH = "research"
EXPLORE = "explore"
QUERY = "query"
VALID_TYPES = {RESEARCH, EXPLORE, QUERY}


def _ensure_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _read_job(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _write_job(job: dict[str, Any]) -> None:
    _ensure_dirs()
    path = _job_path(job["id"])
    path.write_text(json.dumps(job, indent=2))


# ── Public API ────────────────────────────────────────────────────────


def add(job_type: str, subject: str) -> dict[str, Any]:
    """Create a new queued job. Returns the job dict."""
    if job_type not in VALID_TYPES:
        raise ValueError(f"Invalid job type {job_type!r}, must be one of {VALID_TYPES}")

    now = time.time()
    # ID: timestamp + short slug for human readability
    slug = subject.lower()[:40].strip()
    slug = "".join(c if c.isalnum() else "-" for c in slug)
    slug = slug.strip("-") or "job"
    job_id = f"{int(now)}-{slug}"

    job = {
        "id": job_id,
        "type": job_type,
        "subject": subject,
        "status": QUEUED,
        "created_at": now,
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result_dir": None,
    }
    _write_job(job)
    return job


def next_job() -> dict[str, Any] | None:
    """Return the oldest queued job, or None if the queue is empty."""
    _ensure_dirs()
    queued = []
    for path in JOBS_DIR.glob("*.json"):
        job = _read_job(path)
        if job and job.get("status") == QUEUED:
            queued.append(job)

    if not queued:
        return None

    # Sort by created_at (oldest first)
    queued.sort(key=lambda j: j["created_at"])
    return queued[0]


def mark_running(job_id: str) -> dict[str, Any]:
    """Mark a job as running."""
    path = _job_path(job_id)
    job = _read_job(path)
    if not job:
        raise ValueError(f"Job {job_id} not found")
    job["status"] = RUNNING
    job["started_at"] = time.time()
    _write_job(job)
    return job


def mark_done(job_id: str, result_dir: str | None = None) -> dict[str, Any]:
    """Mark a job as completed."""
    path = _job_path(job_id)
    job = _read_job(path)
    if not job:
        raise ValueError(f"Job {job_id} not found")
    job["status"] = DONE
    job["completed_at"] = time.time()
    job["result_dir"] = result_dir
    _write_job(job)
    return job


def mark_failed(job_id: str, error: str) -> dict[str, Any]:
    """Mark a job as failed."""
    path = _job_path(job_id)
    job = _read_job(path)
    if not job:
        raise ValueError(f"Job {job_id} not found")
    job["status"] = FAILED
    job["completed_at"] = time.time()
    job["error"] = error
    _write_job(job)
    return job


def list_jobs(status: str | None = None) -> list[dict[str, Any]]:
    """List all jobs, optionally filtered by status. Most recent first."""
    _ensure_dirs()
    jobs = []
    for path in JOBS_DIR.glob("*.json"):
        job = _read_job(path)
        if job:
            if status is None or job.get("status") == status:
                jobs.append(job)

    jobs.sort(key=lambda j: j["created_at"], reverse=True)
    return jobs


def queue_size() -> int:
    """Number of queued (pending) jobs."""
    return len(list_jobs(status=QUEUED))


def clear_done() -> int:
    """Remove completed/failed job files. Returns count removed."""
    _ensure_dirs()
    removed = 0
    for path in JOBS_DIR.glob("*.json"):
        job = _read_job(path)
        if job and job.get("status") in (DONE, FAILED):
            path.unlink()
            removed += 1
    return removed
