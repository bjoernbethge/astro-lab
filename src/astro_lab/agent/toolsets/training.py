"""Training helpers: CLI suggestions and human-in-the-loop background jobs."""

from __future__ import annotations

import json
import math
import os
import sys

from pydantic_ai import FunctionToolset

from .._surveys import normalize_survey, survey_keys
from ..training_jobs import (
    delete_training_proposal,
    job_log_tail,
    job_status,
    list_jobs,
    load_training_proposal,
    resolve_config_yaml_under_project,
    save_training_proposal,
    start_training_job,
)

training_toolset = FunctionToolset()

_TRAIN_TASKS = frozenset(
    {
        "node_classification",
        "graph_classification",
        "node_regression",
        "graph_regression",
    }
)
_TRAIN_MODELS = frozenset(
    {
        "gcn",
        "gat",
        "sage",
        "gin",
        "transformer",
        "pointnet",
        "temporal",
        "auto",
    }
)


def _require_survey(survey: str) -> tuple[str | None, str | None]:
    canon = normalize_survey(survey)
    if canon is None:
        return None, f"Unknown survey '{survey}'. Known: {', '.join(survey_keys())}"
    return canon, None


def _build_validated_train_argv(
    survey: str,
    task: str,
    model: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_samples: int | None,
    config_yaml: str | None,
) -> tuple[list[str], dict[str, object]] | str:
    canon, err = _require_survey(survey)
    if err:
        return json.dumps({"error": err})
    t = task.strip()
    if t not in _TRAIN_TASKS:
        return json.dumps({"error": "invalid_task", "allowed": sorted(_TRAIN_TASKS)})
    m = model.strip()
    if m not in _TRAIN_MODELS:
        return json.dumps({"error": "invalid_model", "allowed": sorted(_TRAIN_MODELS)})
    try:
        ep = max(1, min(int(epochs), 500_000))
        bs = max(1, min(int(batch_size), 8192))
        lr = float(learning_rate)
        if not (math.isfinite(lr) and 0.0 < lr < 1000.0):
            return json.dumps({"error": "invalid_learning_rate"})
    except (TypeError, ValueError):
        return json.dumps({"error": "invalid_numeric_parameters"})

    ms: int | None = None
    if max_samples is not None:
        try:
            ms = max(1, min(int(max_samples), 50_000_000))
        except (TypeError, ValueError):
            return json.dumps({"error": "invalid_max_samples"})

    if config_yaml:
        cfg_path = resolve_config_yaml_under_project(config_yaml.strip())
        if cfg_path is None:
            return json.dumps(
                {
                    "error": "config_yaml_not_found_or_outside_project",
                    "path": config_yaml,
                },
            )
        cfg_arg = str(cfg_path)
    else:
        cfg_arg = None

    argv = [
        sys.executable,
        "-m",
        "astro_lab.cli",
        "train",
        canon,
        "--task",
        t,
        "--model",
        m,
        "--epochs",
        str(ep),
        "--batch-size",
        str(bs),
        "--learning-rate",
        str(lr),
    ]
    if ms is not None:
        argv.extend(["--max-samples", str(ms)])
    if cfg_arg:
        argv.extend(["-c", cfg_arg])

    meta: dict[str, object] = {
        "survey": canon,
        "task": t,
        "model": m,
        "epochs": ep,
        "batch_size": bs,
        "learning_rate": lr,
        "max_samples": ms,
        "config_yaml": cfg_arg,
    }
    return argv, meta


def _human_approves_training(meta: dict[str, object]) -> bool:
    if os.environ.get("ASTROLAB_TRAINING_AUTO_APPROVE", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return True
    if not sys.stdin.isatty():
        return False
    summary = (
        f"survey={meta.get('survey')} task={meta.get('task')} model={meta.get('model')} "
        f"epochs={meta.get('epochs')} batch_size={meta.get('batch_size')} "
        f"lr={meta.get('learning_rate')}"
    )
    if meta.get("max_samples") is not None:
        summary += f" max_samples={meta.get('max_samples')}"
    if meta.get("config_yaml"):
        summary += f" config={meta.get('config_yaml')}"
    try:
        reply = input(
            f"\n[AstroLab] Background training — {summary}\n"
            f"Start this job? Type y/yes to confirm [y/N] "
        ).strip().lower()
    except EOFError:
        return False
    return reply in ("y", "yes")


@training_toolset.tool
def suggest_train_cmd(
    survey: str,
    task: str = "node_classification",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    model: str = "auto",
    max_samples: int | None = None,
) -> str:
    """Suggest astro-lab train command for a survey."""
    canon, err = _require_survey(survey)
    if err:
        return err
    parts = [
        "astro-lab",
        "train",
        canon,
        "--task",
        task,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--learning-rate",
        str(learning_rate),
        "--model",
        model,
    ]
    if max_samples is not None:
        parts.extend(["--max-samples", str(max_samples)])
    return " ".join(parts)


@training_toolset.tool
def suggest_optimize_cmd(
    survey: str,
    task: str = "node_classification",
    trials: int = 50,
    max_epochs: int = 20,
) -> str:
    """Suggest astro-lab optimize (HPO) command."""
    canon, err = _require_survey(survey)
    if err:
        return err
    return (
        f"astro-lab optimize {canon} --task {task} "
        f"--trials {trials} --max-epochs {max_epochs}"
    )


@training_toolset.tool
def train_prepare_background(
    survey: str,
    task: str = "node_classification",
    model: str = "auto",
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    max_samples: int | None = None,
    config_yaml: str | None = None,
) -> str:
    """Stage a training plan (human-in-the-loop). Does not start the process. Returns ``proposal_id``."""
    built = _build_validated_train_argv(
        survey, task, model, epochs, batch_size, learning_rate, max_samples, config_yaml
    )
    if isinstance(built, str):
        return built
    argv, meta = built
    proposal_id = save_training_proposal(argv, meta)
    return json.dumps(
        {
            "proposal_id": proposal_id,
            "human_in_the_loop": True,
            "next_step": (
                "Call train_execute_background with this proposal_id after the user confirms "
                "(in the REPL you will get an interactive y/N prompt; non-TTY requires "
                "ASTROLAB_TRAINING_AUTO_APPROVE=1)."
            ),
            "plan": meta,
            "argv_preview": " ".join(argv[4:]),
        },
        indent=2,
    )


@training_toolset.tool
def train_execute_background(proposal_id: str) -> str:
    """Run a staged plan after human confirmation (TTY prompt or ASTROLAB_TRAINING_AUTO_APPROVE)."""
    data = load_training_proposal(proposal_id)
    if data is None:
        return json.dumps(
            {
                "error": "invalid_expired_or_unknown_proposal",
                "proposal_id": proposal_id.strip(),
                "hint": "Create a new plan with train_prepare_background.",
            },
            indent=2,
        )
    argv = data.get("argv")
    meta = data.get("meta")
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        return json.dumps({"error": "corrupt_proposal_argv"}, indent=2)
    if not isinstance(meta, dict):
        return json.dumps({"error": "corrupt_proposal_meta"}, indent=2)

    if not _human_approves_training(meta):
        return json.dumps(
            {
                "error": "not_confirmed",
                "proposal_id": data.get("proposal_id"),
                "hint": (
                    "User declined, non-interactive stdin, or missing ASTROLAB_TRAINING_AUTO_APPROVE. "
                    "Proposal is still valid until it expires."
                ),
            },
            indent=2,
        )

    try:
        job_id, log_path, meta_path = start_training_job(argv, meta)
        written = json.loads(meta_path.read_text(encoding="utf-8"))
        pid_out = written.get("pid")
    except OSError as e:
        return json.dumps({"error": "spawn_failed", "detail": str(e)}, indent=2)

    delete_training_proposal(str(data.get("proposal_id") or proposal_id))
    return json.dumps(
        {
            "job_id": job_id,
            "log_path": str(log_path),
            "meta_path": str(meta_path),
            "pid": pid_out,
            "hint": "Poll with train_job_status / train_job_log_tail; list with train_jobs_list.",
        },
        indent=2,
    )


@training_toolset.tool
def train_discard_background_proposal(proposal_id: str) -> str:
    """Remove a pending training plan without running it."""
    ok = delete_training_proposal(proposal_id)
    return json.dumps(
        {"discarded": ok, "proposal_id": proposal_id.strip()},
        indent=2,
    )


@training_toolset.tool
def train_job_status(job_id: str) -> str:
    """Check whether a background training job process is still running (+ log outcome hints)."""
    return json.dumps(job_status(job_id.strip()), indent=2, default=str)


@training_toolset.tool
def train_job_log_tail(job_id: str, max_lines: int = 40) -> str:
    """Last lines of a job log (for monitoring)."""
    try:
        n = max(1, min(int(max_lines), 500))
    except (TypeError, ValueError):
        n = 40
    return json.dumps(job_log_tail(job_id.strip(), max_lines=n), indent=2, default=str)


@training_toolset.tool
def train_jobs_list(limit: int = 15) -> str:
    """Recent background training jobs (newest first)."""
    try:
        lim = max(1, min(int(limit), 100))
    except (TypeError, ValueError):
        lim = 15
    return json.dumps(list_jobs(limit=lim), indent=2, default=str)
