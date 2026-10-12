"""Run a fixed set of questions through the chat agent and record each
answer with its timing, as a baseline to compare later changes against.

Questions are read from a JSON list (default data/processed/eval/
questions.json, git-ignored since they come from real visitors): each item
has an "id" and a "question", optionally a "fen" (asked about that board
position) and "follow_up_to" (the id of a question asked first in the same
conversation). Results are appended as one JSON line per run, and a summary
table is written next to them.

Timings are measured locally, so engine speed differs from the deployed
app; compare runs made on the same machine.

Run as a module from the repo root:
    .venv/bin/python -m scripts.run_eval_baseline --runs 2
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import anthropic

from src.agent.chess_agent import ask
from src.config import CLAUDE_MODEL
from src.embeddings.voyage_embedder import get_voyage_client
from src.engine.engine_pool import EnginePool
from src.engine.stockfish_eval import get_engine_path
from src.ingestion.db_loader import get_connection_pool
from src.ui.chat import _to_model_text

EVAL_DIR = Path("data/processed/eval")


class _TurnRecorder(logging.Handler):
    """Collects chess_agent's per-turn log lines ("turn N: Xs, tools=[...]")
    for whichever thread is running a question."""

    def __init__(self) -> None:
        super().__init__()
        self.turns: dict[int, list[dict]] = {}

    def emit(self, record: logging.LogRecord) -> None:
        if record.msg.startswith("turn %d:"):
            _, seconds, tools = record.args
            self.turns.setdefault(record.thread, []).append(
                {"seconds": round(seconds, 2), "tools": tools}
            )


def _run_one(item, history, pools, engines, voyage, client, recorder) -> tuple[dict, str]:
    """Ask one question; return its record and the answer text."""
    engine_pool = engines.get()
    thread = threading.get_ident()
    recorder.turns[thread] = []
    started = time.monotonic()
    first_token: list[float] = []
    status_lines: list[tuple[float, str]] = []
    chunk_times: list[float] = []

    def on_chunk(_delta: str) -> None:
        now = time.monotonic() - started
        chunk_times.append(now)
        if not first_token:
            first_token.append(now)

    def on_step(text: str) -> None:
        status_lines.append((time.monotonic() - started, text))

    record = {
        "id": item["id"],
        "question": item["question"],
        "fen": item.get("fen"),
        "follow_up_to": item.get("follow_up_to"),
        "model": CLAUDE_MODEL,
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    answer = ""
    try:
        answer = ask(
            _to_model_text(item["question"], item.get("fen")),
            pools,
            engine_pool,
            voyage,
            client=client,
            on_chunk=on_chunk,
            on_step=on_step,
            history=history,
        )
        record["error"] = None
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        engines.put(engine_pool)
    total = time.monotonic() - started
    last_step = status_lines[-1][0] if status_lines else 0.0
    answer_start = next((t for t in chunk_times if t > last_step), None)
    record.update(
        {
            "answer": answer,
            "answer_chars": len(answer),
            "total_seconds": round(total, 2),
            "first_token_seconds": round(first_token[0], 2) if first_token else None,
            "answer_start_seconds": round(answer_start, 2) if answer_start is not None else None,
            "turns": recorder.turns.pop(thread, []),
            "status_lines": [{"seconds": round(t, 2), "text": s} for t, s in status_lines],
        }
    )
    record["turn_count"] = len(record["turns"])
    record["tools_used"] = sorted({tool for turn in record["turns"] for tool in turn["tools"]})
    return record, answer


def _summary(records: list[dict]) -> str:
    rows = [
        "| id | runs | mean s | min s | max s | turns | chars | errors |",
        "|---|---|---|---|---|---|---|---|",
    ]
    by_id: dict[str, list[dict]] = {}
    for r in records:
        by_id.setdefault(r["id"], []).append(r)
    for qid, runs in by_id.items():
        times = [r["total_seconds"] for r in runs]
        rows.append(
            f"| {qid} | {len(runs)} | {statistics.mean(times):.1f} | {min(times):.1f} | "
            f"{max(times):.1f} | {'/'.join(str(r['turn_count']) for r in runs)} | "
            f"{'/'.join(str(r['answer_chars']) for r in runs)} | "
            f"{sum(1 for r in runs if r['error'])} |"
        )
    all_times = [r["total_seconds"] for r in records if not r["error"]]
    if all_times:
        rows.append("")
        rows.append(
            f"Overall: {len(records)} runs, median {statistics.median(all_times):.1f}s, "
            f"mean {statistics.mean(all_times):.1f}s, slowest {max(all_times):.1f}s."
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.json")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--label", default=datetime.now().strftime("baseline_%Y%m%d_%H%M"))
    args = parser.parse_args()

    items = json.loads(args.questions.read_text(encoding="utf-8"))
    follow_ups = {i["follow_up_to"]: i for i in items if i.get("follow_up_to")}
    chains = [
        [i] + ([follow_ups[i["id"]]] if i["id"] in follow_ups else [])
        for i in items
        if not i.get("follow_up_to")
    ]

    recorder = _TurnRecorder()
    agent_logger = logging.getLogger("src.agent.chess_agent")
    # INFO, or the per-turn records are dropped before reaching the recorder.
    agent_logger.setLevel(logging.INFO)
    agent_logger.addHandler(recorder)
    pools = get_connection_pool()
    engines: queue.Queue = queue.Queue()
    for _ in range(args.workers):
        engines.put(EnginePool(get_engine_path(), size=2))
    voyage, client = get_voyage_client(), anthropic.Anthropic()

    out_path = EVAL_DIR / f"{args.label}.jsonl"
    write_lock = threading.Lock()
    records: list[dict] = []

    def run_chain(chain: list[dict], run: int) -> None:
        history: list[dict[str, str]] = []
        for item in chain:
            record, answer = _run_one(
                item, history or None, pools, engines, voyage, client, recorder
            )
            record["run"] = run
            with write_lock:
                records.append(record)
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
                status = (
                    record["error"] or f"{record['total_seconds']}s, {record['turn_count']} turns"
                )
                print(f"[{len(records)}] {item['id']} run {run}: {status}", flush=True)
            history = history + [
                {"role": "user", "content": _to_model_text(item["question"], item.get("fen"))},
                {"role": "assistant", "content": answer},
            ]

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_chain, chain, run)
                for run in range(1, args.runs + 1)
                for chain in chains
            ]
            for future in futures:
                future.result()
    finally:
        while not engines.empty():
            engines.get().close()
        pools.closeall()

    summary_path = EVAL_DIR / f"{args.label}_summary.md"
    summary_path.write_text(_summary(records), encoding="utf-8")
    print(f"Wrote {out_path} and {summary_path}")


if __name__ == "__main__":
    main()
