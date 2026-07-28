from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


K_VALUES = (18, 20, 23, 25, 28, 30)
METHODS = ("muon", "adaptive_wv_wout")


@dataclass(frozen=True)
class Task:
    K: int
    method: str

    def output_dir(self, root: Path) -> Path:
        return root / f"K{self.K}" / self.method


def print_gpu_status() -> None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        print("nvidia-smi not found; GPU status unavailable")
        return
    if result.returncode:
        print(f"nvidia-smi failed: {result.stderr.strip()}")
        return
    print("GPU status before launch:")
    print(result.stdout.rstrip())


def task_command(task: Task, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "muon_sparse_attention.train",
        "--K",
        str(task.K),
        "--method",
        task.method,
        "--steps",
        str(args.steps),
        "--batch-size",
        str(args.batch_size),
        "--eval-size",
        str(args.eval_size),
        "--eval-every",
        str(args.eval_every),
        "--seed",
        str(args.seed),
        "--device",
        "cuda",
        "--output",
        str(task.output_dir(args.output)),
    ]


def run_task(
    task: Task,
    gpu: str,
    args: argparse.Namespace,
) -> tuple[Task, int]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    log_dir = args.output / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"K{task.K}_{task.method}.log"
    with log_path.open("w") as log:
        process = subprocess.run(
            task_command(task, args),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return task, process.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/experiment_202"))
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--jobs-per-gpu", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-size", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_gpu_status()
    all_tasks = [Task(K, method) for K in K_VALUES for method in METHODS]
    missing = [
        task
        for task in all_tasks
        if args.force or not (task.output_dir(args.output) / "metrics.npz").exists()
    ]
    print(f"output: {args.output}")
    print(f"expected={len(all_tasks)} existing={len(all_tasks)-len(missing)} missing={len(missing)}")
    for task in missing:
        print(f"  K={task.K:2d} method={task.method}")
    if args.plan:
        return
    if args.plot_only and missing:
        raise RuntimeError("plot-only requested but one or more runs are missing")

    if not args.plot_only and missing:
        gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
        slots = [
            gpu for gpu in gpus for _ in range(max(1, args.jobs_per_gpu))
        ]
        if not slots:
            raise ValueError("at least one GPU is required")
        with ThreadPoolExecutor(max_workers=len(slots)) as executor:
            futures = {
                executor.submit(run_task, task, slots[index % len(slots)], args): task
                for index, task in enumerate(missing)
            }
            for future in as_completed(futures):
                task, returncode = future.result()
                if returncode:
                    raise RuntimeError(
                        f"K={task.K} {task.method} failed with code {returncode}"
                    )
                print(f"finished K={task.K} {task.method}")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "muon_sparse_attention.plot",
            "--input",
            str(args.output),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
