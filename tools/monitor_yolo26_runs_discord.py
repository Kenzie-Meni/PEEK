#!/usr/bin/env python3
"""Monitor YOLO26 training runs, post Discord updates, then run test evals."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time


RUNS = {
    "mapmax_640_refine": {
        "label": "YOLO26s mAP-max 640 refine",
        "dir": "runs/detect/peek_yolo26s_bbox_mapmax_640_refine_tmux",
        "epochs": 100,
        "session": "yolo26_mapmax_refine",
    },
    "mapmax_640": {
        "label": "YOLO26s mAP-max 640 fresh",
        "dir": "runs/detect/peek_yolo26s_bbox_mapmax_640_fresh_tmux",
        "epochs": 200,
        "session": "yolo26_mapmax_fresh",
    },
    "smallobj_640": {
        "label": "YOLO26s small-object recall 640 tune",
        "dir": "runs/detect/peek_yolo26s_bbox_smallobj_recall_640_tmux",
        "epochs": 120,
        "session": "yolo26_smallobj_tune",
    },
    "smallobj_balanced_640": {
        "label": "YOLO26s balanced small-object 640 tune",
        "dir": "runs/detect/peek_yolo26s_bbox_smallobj_balanced_640_tmux",
        "epochs": 100,
        "session": "yolo26_smallobj_balanced",
    },
    "recall_recover_640": {
        "label": "YOLO26s recall-to-precision recovery 640 tune",
        "dir": "runs/detect/peek_yolo26s_bbox_recall_recover_640_tmux",
        "epochs": 80,
        "session": "yolo26_recall_recover",
    },
    "tiny_conservative_640": {
        "label": "YOLO26s conservative tiny-component 640 tune",
        "dir": "runs/detect/peek_yolo26s_bbox_tiny_conservative_640_tmux",
        "epochs": 100,
        "session": "yolo26_tiny_conservative",
    },
    "fresh_lowaug_polish_640": {
        "label": "YOLO26s fresh-best low-augmentation polish 640 tune",
        "dir": "runs/detect/peek_yolo26s_bbox_fresh_lowaug_polish_640_tmux",
        "epochs": 80,
        "session": "yolo26_fresh_lowaug_polish",
    },
    "smallobj_loc_recover_640": {
        "label": "YOLO26s small-object localization recovery 640 tune",
        "dir": "runs/detect/peek_yolo26s_bbox_smallobj_loc_recover_640_tmux",
        "epochs": 80,
        "session": "yolo26_smallobj_loc_recover",
    },
    "fresh_alt_seed_640": {
        "label": "YOLO26s fresh alternate-seed 640 mAP run",
        "dir": "runs/detect/peek_yolo26s_bbox_fresh_alt_seed17_640_tmux",
        "epochs": 200,
        "session": "yolo26_fresh_alt_seed",
    },
    "smallobj_micro_polish_640": {
        "label": "YOLO26s small-object micro-polish 640 tune",
        "dir": "runs/detect/peek_yolo26s_bbox_smallobj_micro_polish_640_tmux",
        "epochs": 40,
        "session": "yolo26_smallobj_micro_polish",
    },
}


EVENT_LOG_PATH: Path | None = None


def log_event(message: str) -> None:
    if EVENT_LOG_PATH is None:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    with EVENT_LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def post_discord(webhook: str, content: str) -> None:
    content = f"fit-afrl: {content}"
    log_event(content)
    for chunk_start in range(0, len(content), 1900):
        chunk = content[chunk_start : chunk_start + 1900]
        data = json.dumps({"content": chunk}).encode("utf-8")
        process = subprocess.run(
            [
                "curl",
                "-fsS",
                "--connect-timeout",
                "10",
                "--max-time",
                "30",
                "-A",
                "fit-afrl-yolo-monitor/1.0",
                "-H",
                "Content-Type: application/json",
                "-d",
                data.decode("utf-8"),
                webhook,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if process.returncode:
            raise RuntimeError(
                f"Discord curl failed with return code {process.returncode}: {process.stderr.strip()}"
            )


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"posted_finished": [], "posted_overall_best": None, "eval_done": False}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def read_metrics(run_dir: Path) -> dict | None:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        return None
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [{key.strip(): value.strip() for key, value in row.items()} for row in reader]
    if not rows:
        return None

    def f(row: dict, key: str) -> float:
        return float(row[key])

    last = rows[-1]
    best = max(
        rows,
        key=lambda row: (
            f(row, "metrics/mAP50(B)"),
            f(row, "metrics/mAP50-95(B)"),
        ),
    )
    return {
        "last_epoch": int(float(last["epoch"])),
        "best_epoch": int(float(best["epoch"])),
        "precision": f(best, "metrics/precision(B)"),
        "recall": f(best, "metrics/recall(B)"),
        "map50": f(best, "metrics/mAP50(B)"),
        "map5095": f(best, "metrics/mAP50-95(B)"),
    }


def tmux_session_exists(session: str) -> bool:
    process = subprocess.run(
        ["tmux", "has-session", "-t", session],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return process.returncode == 0


def run_finished(run_dir: Path, expected_epochs: int, session: str | None = None) -> bool:
    train_log = run_dir / "train.log"
    metrics = read_metrics(run_dir)
    if metrics and metrics["last_epoch"] >= expected_epochs:
        return True
    if train_log.exists():
        tail = train_log.read_text(encoding="utf-8", errors="replace")[-20000:]
        return "Training complete" in tail or "Results saved to" in tail and "Validating" in tail
    if metrics and session and not tmux_session_exists(session):
        return True
    return False


def format_run_message(name: str, info: dict, metrics: dict, finished: list[str]) -> str:
    remaining = [RUNS[key]["label"] for key in RUNS if key not in finished and key != name]
    if remaining:
        next_text = "Next: still waiting on " + ", ".join(remaining) + "."
    else:
        next_text = "Next: all trainings are finished; starting separate test-set evaluation."
    return (
        f"{info['label']} finished.\n"
        f"Best val epoch: {metrics['best_epoch']}\n"
        f"mAP50: {metrics['map50']:.5f}\n"
        f"mAP50-95: {metrics['map5095']:.5f}\n"
        f"Precision: {metrics['precision']:.5f}\n"
        f"Recall: {metrics['recall']:.5f}\n"
        f"{next_text}"
    )


def format_best_message(name: str, metrics: dict, initial: bool) -> str:
    prefix = "Current overall best" if initial else "New overall validation best"
    return (
        f"{prefix}: {RUNS[name]['label']}\n"
        f"Epoch: {metrics['best_epoch']}\n"
        f"mAP50: {metrics['map50']:.5f}\n"
        f"mAP50-95: {metrics['map5095']:.5f}\n"
        f"Precision: {metrics['precision']:.5f}\n"
        f"Recall: {metrics['recall']:.5f}"
    )


def metric_tuple(metrics: dict) -> tuple[float, float, int]:
    return (metrics["map50"], metrics["map5095"], metrics["best_epoch"])


def state_metric_tuple(best: dict | None) -> tuple[float, float, int]:
    if not best:
        return (-1.0, -1.0, -1)
    return (float(best["map50"]), float(best["map5095"]), int(best["best_epoch"]))


def evaluate_best(repo: Path, winner: str, weights: Path, webhook: str, device: str, batch: int) -> str:
    post_discord(
        webhook,
        f"Both YOLO26s training runs are done. Winner by validation mAP50: {RUNS[winner]['label']}. "
        "Running separate test-set evaluation now at imgsz 640.",
    )
    log_path = repo / "runs" / "detect" / "eval_tests" / "discord_eval.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(repo / "tools" / "eval_yolo26_tests.py"),
        "--weights",
        str(weights),
        "--imgsz",
        "640",
        "--device",
        device,
        "--batch",
        str(batch),
    ]
    process = subprocess.run(
        command,
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(process.stdout, encoding="utf-8")
    if process.returncode:
        message = (
            "Separate test-set evaluation failed.\n"
            f"Command return code: {process.returncode}\n"
            f"Log: {log_path}"
        )
        post_discord(webhook, message)
        raise RuntimeError(message)

    lines = [
        line
        for line in process.stdout.splitlines()
        if " imgsz=" in line and "mAP50=" in line
    ]
    summary = "\n".join(lines) if lines else process.stdout[-1500:]
    message = (
        f"Separate test-set evaluation complete for {RUNS[winner]['label']}.\n"
        f"Weights: {weights}\n"
        f"{summary}\n"
        "Done."
    )
    post_discord(webhook, message)
    return summary


def main() -> None:
    global EVENT_LOG_PATH

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--interval", type=int, default=120)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--state", default=None)
    args = parser.parse_args()

    webhook = os.environ.get("DISCORD_WEBHOOK")
    if not webhook:
        raise SystemExit("DISCORD_WEBHOOK is required")

    repo = Path(args.repo).resolve()
    EVENT_LOG_PATH = repo / "runs" / "detect" / "discord_monitor_events.log"
    state_path = Path(args.state) if args.state else repo / "runs" / "detect" / "discord_monitor_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = load_state(state_path)
    post_discord(webhook, "YOLO26s monitor is active. I will post when each run finishes, then evaluate separate test sets.")

    while True:
        state = load_state(state_path)
        posted = set(state.get("posted_finished", []))
        state.pop("posted_best", None)
        posted_overall_best = state.get("posted_overall_best")
        finished = []
        metrics_by_run = {}
        for name, info in RUNS.items():
            run_dir = repo / info["dir"]
            metrics = read_metrics(run_dir)
            if metrics:
                metrics_by_run[name] = metrics
                if metric_tuple(metrics) > state_metric_tuple(posted_overall_best):
                    post_discord(webhook, format_best_message(name, metrics, initial=posted_overall_best is None))
                    posted_overall_best = {
                        "run": name,
                        "best_epoch": metrics["best_epoch"],
                        "map50": metrics["map50"],
                        "map5095": metrics["map5095"],
                    }
                    state["posted_overall_best"] = posted_overall_best
                    save_state(state_path, state)
            if run_finished(run_dir, int(info["epochs"]), info.get("session")):
                finished.append(name)
                if name not in posted and metrics:
                    post_discord(webhook, format_run_message(name, info, metrics, finished))
                    posted.add(name)
                    state["posted_finished"] = sorted(posted)
                    save_state(state_path, state)

        if len(finished) == len(RUNS) and not state.get("eval_done"):
            winner = max(
                finished,
                key=lambda name: (
                    metrics_by_run[name]["map50"],
                    metrics_by_run[name]["map5095"],
                ),
            )
            winner_weights = repo / RUNS[winner]["dir"] / "weights" / "best.pt"
            summary = evaluate_best(repo, winner, winner_weights, webhook, args.device, args.batch)
            state["eval_done"] = True
            state["winner"] = winner
            state["eval_summary"] = summary
            save_state(state_path, state)
            return

        save_state(state_path, state)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
