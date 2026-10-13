"""Resumable sequential driver for the completion runs on a shared workstation.

Reads a JSON job list and runs the jobs in order. A job is skipped when its `done` file
exists, so the driver can be restarted after an interruption. GPU jobs go through the
workstation's FIFO queue (one ticket per job, so other projects can interleave); CPU jobs
start only when enough RAM is free. A job may name `requires` files that must exist first.
A failed job stops the driver: nothing is retried silently.

Job fields: name, cmd (list), kind ("gpu" | "cpu"), done (path), requires (list of paths),
vram_gb, ram_gb, env (dict).
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

GPUQ = "D:/PROJECTS/.cognition/gpu_queue/gpuq.py"


def free_ram_gb():
    try:
        import psutil
        return psutil.virtual_memory().available/2**30
    except ImportError:
        return 99.0


def log(stream, text):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {text}"
    print(line, flush=True)
    stream.write(line + "\n")
    stream.flush()


def run(jobs_path, log_dir, poll):
    jobs = json.loads(Path(jobs_path).read_text(encoding="utf-8"))
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir/"driver.log").open("a", encoding="utf-8") as journal:
        log(journal, f"driver start: {jobs_path} ({len(jobs)} jobs)")
        for job in jobs:
            if Path(job["done"]).exists():
                continue
            for required in job.get("requires", []):
                while not Path(required).exists():
                    time.sleep(poll)
            if job["kind"] == "cpu":
                while free_ram_gb() < job.get("ram_gb", 3):
                    time.sleep(poll)
                cmd = job["cmd"]
            else:
                cmd = [sys.executable, GPUQ, "run", "--name", f"neuro3d-lab:{job['name']}",
                       "--vram", str(job.get("vram_gb", 4)), "--ram", str(job.get("ram_gb", 3)),
                       "--cwd", os.getcwd(), "--"] + job["cmd"]
            env = dict(os.environ, PYTHONUNBUFFERED="1", CUBLAS_WORKSPACE_CONFIG=":4096:8", **job.get("env", {}))
            log(journal, f"start {job['name']}")
            t0 = time.time()
            with (log_dir/f"{job['name']}.log").open("a", encoding="utf-8") as out:
                code = subprocess.run(cmd, stdout=out, stderr=subprocess.STDOUT, env=env).returncode
            log(journal, f"end {job['name']} rc={code} {time.time() - t0:.0f}s")
            if code != 0 or not Path(job["done"]).exists():
                log(journal, f"STOP: {job['name']} failed or left no {job['done']}")
                return 1
        log(journal, "driver finished: every job done")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", type=Path)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--poll", type=float, default=60)
    args = parser.parse_args()
    raise SystemExit(run(args.jobs, args.logs, args.poll))


if __name__ == "__main__":
    main()
