#!/usr/bin/env python3
"""Launch the paired natural-task conflict experiment across idle GPUs."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
TASKS = ("openwebmath", "finemath4plus", "proofpile2", "stackmathqa")
METHODS = {
    "adam": ("pythia_adam", "none"),
    "muon": ("matched", "0:muon5ns:0:0:0"),
    "pion": ("matched", "0:pion:3:2:.45"),
}
STEPS = "0,1,5,10,20,50,100,200,400,610"


def available_gpus() -> list[int]:
    data = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True
    )
    free = []
    for line in data.splitlines():
        index, used, total, util = (int(part.strip()) for part in line.split(","))
        if used < 2048 and util < 25 and total - used > 16000:
            free.append(index)
    if not free:
        raise RuntimeError("No sufficiently idle GPU is available")
    return free


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--tasks", default=",".join(TASKS),
                        help="Comma-separated task cache names; default preserves original study")
    parser.add_argument("--smoke-all", action="store_true",
                        help="Smoke every selected task/method instead of the original first task")
    args = parser.parse_args()
    tasks = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("--tasks must specify unique task names")
    root = args.run_root.resolve()
    cache = root / "cache"
    if not cache.is_dir():
        raise FileNotFoundError(cache)
    phase = "smoke" if args.smoke else "training"
    phase_dir = root / phase
    if phase_dir.exists() and any(phase_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {phase_dir}")
    phase_dir.mkdir(parents=True, exist_ok=True)
    gpus = available_gpus()
    rows = []
    jobs = ([(task, method) for task in tasks for method in METHODS]
            if (not args.smoke or args.smoke_all) else
            [(tasks[0], method) for method in METHODS])
    (phase_dir / "gpu_preflight.json").write_text(json.dumps(gpus) + "\n")
    active: dict[int, tuple[subprocess.Popen, dict, object]] = {}
    pending = iter(jobs)

    def launch(gpu: int, task: str, method: str) -> None:
        family, schedule = METHODS[method]
        name = f"{task}_{method}_seed42"
        output = phase_dir / name
        log_path = phase_dir / f"{name}.log"
        command = [sys.executable, str(HERE / "train_openwebmath.py"),
                   "--cache-dir", str(cache / task), "--output-dir", str(output),
                   "--seed", "42", "--rho-label", "rho20",
                   "--optimizer-family", family, "--spectral-schedule", schedule,
                   "--total-updates", "610", "--stop-after", "2" if args.smoke else "610",
                   "--device-batch-size", "8", "--grad-accum-steps", "4",
                   "--eval-every", "1" if args.smoke else "10",
                   "--monitor-rows", "16" if args.smoke else "32",
                   "--monitor-retention-every-validation",
                   "--full-validation-steps", "0,2" if args.smoke else "0,610",
                   "--retention-steps", "" if args.smoke else "0,610",
                   "--save-steps", "0,1,2" if args.smoke else STEPS,
                   "--diagnostic-updates", "0,1" if args.smoke else "0"]
        if args.smoke:
            command.append("--smoke")
        row = dict(task=task, method=method, seed=42, gpu=gpu, status="running",
                   output=str(output), log=str(log_path), command=command,
                   start=time.time())
        rows.append(row)
        handle = log_path.open("w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
        active[gpu] = (process, row, handle)
        print(f"START {name} GPU{gpu} PID{process.pid}", flush=True)

    for gpu in gpus:
        try:
            task, method = next(pending)
        except StopIteration:
            break
        launch(gpu, task, method)

    failed = False
    while active:
        time.sleep(5)
        for gpu, (process, row, handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            row["end"] = time.time()
            row["exit_code"] = code
            row["status"] = "complete" if code == 0 else "failed"
            print(f"{row['status'].upper()} {row['task']}_{row['method']} GPU{gpu}", flush=True)
            if code:
                failed = True
            del active[gpu]
            try:
                task, method = next(pending)
            except StopIteration:
                continue
            launch(gpu, task, method)
        (phase_dir / "jobs.json").write_text(json.dumps(rows, indent=2) + "\n")
    (phase_dir / "jobs.json").write_text(json.dumps(rows, indent=2) + "\n")
    if failed:
        raise SystemExit(1)
    print(f"PHASE_COMPLETE {phase_dir}", flush=True)


if __name__ == "__main__":
    main()
