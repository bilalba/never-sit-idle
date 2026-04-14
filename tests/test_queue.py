"""Tests for the file-based job queue."""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import patch

from agent import queue as Q


@pytest.fixture(autouse=True)
def isolated_queue(tmp_path, monkeypatch):
    """Point the queue at a temp directory for every test."""
    jobs_dir = tmp_path / "jobs"
    monkeypatch.setattr(Q, "QUEUE_DIR", tmp_path)
    monkeypatch.setattr(Q, "JOBS_DIR", jobs_dir)


class TestAdd:
    def test_creates_job_file(self):
        job = Q.add("research", "React hooks")
        assert job["type"] == "research"
        assert job["subject"] == "React hooks"
        assert job["status"] == Q.QUEUED
        assert job["created_at"] > 0
        assert job["started_at"] is None

        # File exists
        path = Q._job_path(job["id"])
        assert path.exists()
        stored = json.loads(path.read_text())
        assert stored["id"] == job["id"]

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid job type"):
            Q.add("invalid", "topic")

    def test_slug_from_subject(self):
        job = Q.add("research", "WebSocket performance!")
        assert "websocket-performance" in job["id"]

    def test_empty_subject_slug(self):
        job = Q.add("research", "   ")
        assert "job" in job["id"]


class TestNextJob:
    def test_returns_none_when_empty(self):
        assert Q.next_job() is None

    def test_returns_oldest_first(self):
        j1 = Q.add("research", "first")
        # Bump time so ordering is clear
        j2_data = Q.add("research", "second")

        got = Q.next_job()
        assert got["id"] == j1["id"]

    def test_skips_non_queued(self):
        j1 = Q.add("research", "will be running")
        Q.mark_running(j1["id"])

        j2 = Q.add("research", "still queued")

        got = Q.next_job()
        assert got["id"] == j2["id"]


class TestStatusTransitions:
    def test_mark_running(self):
        job = Q.add("research", "topic")
        updated = Q.mark_running(job["id"])
        assert updated["status"] == Q.RUNNING
        assert updated["started_at"] is not None

    def test_mark_done(self):
        job = Q.add("research", "topic")
        Q.mark_running(job["id"])
        updated = Q.mark_done(job["id"], result_dir="research-topic")
        assert updated["status"] == Q.DONE
        assert updated["completed_at"] is not None
        assert updated["result_dir"] == "research-topic"

    def test_mark_failed(self):
        job = Q.add("research", "topic")
        Q.mark_running(job["id"])
        updated = Q.mark_failed(job["id"], error="LLM timeout")
        assert updated["status"] == Q.FAILED
        assert updated["error"] == "LLM timeout"

    def test_mark_missing_job_raises(self):
        with pytest.raises(ValueError, match="not found"):
            Q.mark_running("nonexistent-id")


class TestListJobs:
    def test_list_all(self):
        Q.add("research", "a")
        Q.add("explore", "b")
        Q.add("query", "c")
        assert len(Q.list_jobs()) == 3

    def test_filter_by_status(self):
        j1 = Q.add("research", "a")
        Q.add("research", "b")
        Q.mark_running(j1["id"])

        assert len(Q.list_jobs(status=Q.QUEUED)) == 1
        assert len(Q.list_jobs(status=Q.RUNNING)) == 1
        assert len(Q.list_jobs(status=Q.DONE)) == 0

    def test_most_recent_first(self):
        j1 = Q.add("research", "old")
        j2 = Q.add("research", "new")
        jobs = Q.list_jobs()
        assert jobs[0]["id"] == j2["id"]
        assert jobs[1]["id"] == j1["id"]


class TestQueueSize:
    def test_counts_only_queued(self):
        j1 = Q.add("research", "a")
        Q.add("research", "b")
        Q.mark_running(j1["id"])
        assert Q.queue_size() == 1


class TestClearDone:
    def test_removes_done_and_failed(self):
        j1 = Q.add("research", "done-job")
        j2 = Q.add("research", "failed-job")
        j3 = Q.add("research", "still-queued")

        Q.mark_running(j1["id"])
        Q.mark_done(j1["id"])
        Q.mark_running(j2["id"])
        Q.mark_failed(j2["id"], error="oops")

        removed = Q.clear_done()
        assert removed == 2

        remaining = Q.list_jobs()
        assert len(remaining) == 1
        assert remaining[0]["id"] == j3["id"]
