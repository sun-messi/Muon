#!/usr/bin/env python3
"""GPU-parallel, non-overwriting diagnostic launcher."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
TASKS = ("openwebmath", "finemath4plus", "proofpile2", "stackmathqa")
METHODS = ("adam", "muon", "pion")


def gpus():
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True)
    return [i for line in output.splitlines()
            for i, used, total, util in [[int(x.strip()) for x in line.split(",")]]
            if used < 2048 and util < 25 and total-used > 16000]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--tasks", default=",".join(TASKS),
                        help="Comma-separated task names; default preserves original study")
    parser.add_argument("--phase", choices=("gradients", "one_step", "fixed"), required=True)
    parser.add_argument("--available-only", action="store_true",
                        help="Queue only routes with a completed training checkpoint.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip diagnostic outputs that already contain the expected CSV.")
    parser.add_argument("--tag", default="all")
    parser.add_argument("--fixed-reference-root", type=Path,
                        help="Use OpenWebMath Muon checkpoints from this run for fixed probes; default is --run-root.")
    args = parser.parse_args()
    tasks = tuple(item.strip() for item in args.tasks.split(",") if item.strip())
    if not tasks or len(tasks) != len(set(tasks)):
        raise ValueError("--tasks must specify unique task names")
    root = args.run_root.resolve()
    phase_root = root / "diagnostics" / args.phase
    if phase_root.exists() and any(phase_root.iterdir()) and not args.resume:
        raise FileExistsError(phase_root)
    phase_root.mkdir(parents=True, exist_ok=True)
    devices = gpus()
    if not devices:
        raise RuntimeError("No free GPU for diagnostics")
    jobs = []
    if args.phase == "fixed":
        reference_root = args.fixed_reference_root.resolve() if args.fixed_reference_root else root
        for step in (0, 200):
            reference = reference_root / "training/openwebmath_muon_seed42/checkpoints" / f"state_step{step:06d}.pt"
            for task in tasks:
                name = f"fixed_step{step}_{task}"
                output = phase_root / name
                if args.resume and (output / "checkpoints.json").exists() and (output / "gradient_cosine_raw.csv").exists():
                    continue
                command = [sys.executable, str(HERE / "analyze_real_task_conflict_gradients.py"),
                           "--run-root", str(root), "--task", task, "--method", "muon",
                           "--output-dir", str(output), "--steps", str(step),
                           "--fixed-checkpoint", str(reference)]
                jobs.append((name, command))
    else:
        script = ("analyze_real_task_conflict_gradients.py" if args.phase == "gradients"
                  else "analyze_real_task_conflict_one_step.py")
        for task in tasks:
            for method in METHODS:
                name = f"{task}_{method}_seed42"
                output = phase_root / name
                if args.available_only and not (root / "training" / name / "complete.json").exists():
                    continue
                expected = "gradient_cosine_raw.csv" if args.phase == "gradients" else "one_step.csv"
                if args.resume and (output / expected).exists():
                    continue
                if output.exists():
                    raise FileExistsError(f"incomplete output requires a separate retry: {output}")
                command = [sys.executable, str(HERE / script), "--run-root", str(root),
                           "--task", task, "--method", method, "--output-dir", str(output)]
                jobs.append((name, command))
    active = {}
    records = []
    remaining = iter(jobs)

    def start(gpu, name, command):
        log = phase_root / f"{name}.log"
        handle = log.open("w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
        row = dict(name=name, gpu=gpu, pid=process.pid, status="running", command=command,
                   log=str(log), started=time.time())
        records.append(row)
        active[gpu] = (process, row, handle)
        print(f"START {name} GPU{gpu} PID{process.pid}", flush=True)

    for gpu in devices:
        try:
            name, command = next(remaining)
        except StopIteration:
            break
        start(gpu, name, command)
    failed = False
    while active:
        time.sleep(5)
        for gpu, (process, row, handle) in list(active.items()):
            code = process.poll()
            if code is None:
                continue
            handle.close()
            row.update(status="complete" if code == 0 else "failed", exit_code=code,
                       ended=time.time())
            failed |= code != 0
            print(f"{row['status'].upper()} {row['name']} GPU{gpu}", flush=True)
            del active[gpu]
            try:
                name, command = next(remaining)
            except StopIteration:
                continue
            start(gpu, name, command)
        (phase_root / f"jobs_{args.tag}.json").write_text(json.dumps(records, indent=2) + "\n")
    if failed:
        raise SystemExit(1)
    print(f"PHASE_COMPLETE {phase_root}", flush=True)


if __name__ == "__main__":
    main()
