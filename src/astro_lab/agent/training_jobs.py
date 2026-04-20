"""Background training subprocesses for the agent (job dir + log tail + status)."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import psutil

from astro_lab.config import find_project_root


def _jobs_root() -> Path:
    root = find_project_root().resolve()
    d = root / ".astro_lab" / "training_jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _pending_root() -> Path:
    p = _jobs_root() / "pending"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _proposal_id_ok(proposal_id: str) -> bool:
    s = proposal_id.strip()
    return bool(s) and len(s) <= 32 and all(c in "0123456789abcdef" for c in s.lower())


def save_training_proposal(
    argv: Sequence[str],
    meta: dict[str, object],
    *,
    ttl_minutes: int = 45,
) -> str:
    """Write a pending training plan; returns ``proposal_id``."""
    proposal_id = uuid.uuid4().hex[:14]
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=max(5, min(int(ttl_minutes), 24 * 60)))
    path = _pending_root() / f"{proposal_id}.json"
    payload = {
        "proposal_id": proposal_id,
        "created_at": now.isoformat(),
        "expires_at": expires.isoformat(),
        "argv": list(argv),
        "meta": meta,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return proposal_id


def load_training_proposal(proposal_id: str) -> dict[str, object] | None:
    if not _proposal_id_ok(proposal_id):
        return None
    path = _pending_root() / f"{proposal_id.strip().lower()}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    exp_raw = str(data.get("expires_at") or "")
    try:
        exp = datetime.fromisoformat(exp_raw.replace("Z", "+00:00"))
        if exp.tzinfo is None:
            exp = exp.replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    if datetime.now(timezone.utc) > exp:
        return None
    return data


def delete_training_proposal(proposal_id: str) -> bool:
    if not _proposal_id_ok(proposal_id):
        return False
    path = _pending_root() / f"{proposal_id.strip().lower()}.json"
    if not path.is_file():
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False


def resolve_config_yaml_under_project(rel_or_abs: str) -> Path | None:
    """YAML file under project root (must exist)."""
    root = find_project_root().resolve()
    p = Path(rel_or_abs).expanduser()
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.resolve()
    try:
        p.relative_to(root)
    except ValueError:
        return None
    if p.suffix.lower() not in {".yaml", ".yml"}:
        return None
    return p if p.is_file() else None


def start_training_job(argv: Sequence[str], meta: dict[str, object]) -> tuple[str, Path, Path]:
    """Spawn ``argv`` with cwd=project root; stdout/stderr to ``<job_id>.log``. Returns (job_id, log_path, meta_path)."""
    job_id = uuid.uuid4().hex[:12]
    jr = _jobs_root()
    log_path = jr / f"{job_id}.log"
    meta_path = jr / f"{job_id}.meta.json"
    root = find_project_root().resolve()
    creationflags = 0
    if sys.platform == "win32":
        creationflags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))

    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            list(argv),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(root),
            creationflags=creationflags,
        )

    payload = {
        **meta,
        "job_id": job_id,
        "pid": proc.pid,
        "argv": list(argv),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "log_path": str(log_path.resolve()),
        "cwd": str(root),
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return job_id, log_path, meta_path


def load_job_meta(job_id: str) -> dict[str, object] | None:
    p = _jobs_root() / f"{job_id.strip()}.meta.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def job_status(job_id: str) -> dict[str, object]:
    meta = load_job_meta(job_id)
    if meta is None:
        return {"error": "unknown_job", "job_id": job_id.strip()}
    pid = int(meta.get("pid") or 0)
    alive = False
    if pid > 0:
        try:
            alive = psutil.pid_exists(pid)
        except (ValueError, TypeError):
            alive = False
    out: dict[str, object] = {
        "job_id": meta.get("job_id"),
        "pid": pid,
        "running": alive,
        "started_at": meta.get("started_at"),
        "survey": meta.get("survey"),
        "task": meta.get("task"),
        "log_path": meta.get("log_path"),
        "argv": meta.get("argv"),
    }
    if not alive and meta.get("log_path"):
        logp = Path(str(meta["log_path"]))
        if logp.is_file():
            try:
                text = logp.read_text(encoding="utf-8", errors="replace")
                tail = text[-4000:] if len(text) > 4000 else text
                if "Training completed!" in text:
                    out["outcome_hint"] = "completed"
                elif "Training failed" in text or "ERROR" in tail[-500:]:
                    out["outcome_hint"] = "likely_failed_or_error_in_log"
                else:
                    out["outcome_hint"] = "exited_check_log"
            except OSError as e:
                out["log_read_error"] = str(e)
    return out


def job_log_tail(job_id: str, max_lines: int = 50) -> dict[str, object]:
    meta = load_job_meta(job_id)
    if meta is None:
        return {"error": "unknown_job", "job_id": job_id.strip()}
    log_path = Path(str(meta.get("log_path") or ""))
    if not log_path.is_file():
        return {"error": "log_missing", "job_id": job_id.strip(), "path": str(log_path)}
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as e:
        return {"error": str(e), "path": str(log_path)}
    n = max(1, min(int(max_lines), 500))
    tail = lines[-n:]
    return {"job_id": meta.get("job_id"), "path": str(log_path), "lines": len(lines), "tail": tail}


def list_jobs(limit: int = 20) -> dict[str, object]:
    jr = _jobs_root()
    metas = sorted(jr.glob("*.meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    lim = max(1, min(int(limit), 100))
    rows: list[dict[str, object]] = []
    for p in metas[:lim]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            jid = str(data.get("job_id", p.stem))
            pid = int(data.get("pid") or 0)
            running = bool(pid and psutil.pid_exists(pid))
            rows.append(
                {
                    "job_id": jid,
                    "survey": data.get("survey"),
                    "task": data.get("task"),
                    "started_at": data.get("started_at"),
                    "running": running,
                    "pid": pid,
                    "log_path": data.get("log_path"),
                }
            )
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    return {"jobs_dir": str(jr), "n_listed": len(rows), "jobs": rows}
