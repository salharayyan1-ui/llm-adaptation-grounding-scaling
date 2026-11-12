"""Run the independent jobs of a task in parallel on every available GPU (24 GB 3090/4090 class).

A tiny dependency-aware scheduler: each GPU runs one job at a time (``CUDA_VISIBLE_DEVICES`` pinned),
jobs start as soon as their dependencies finished and a GPU is free, heavier jobs first. With one GPU
the same schedule simply runs sequentially, so the command is identical on 1 or 2 GPUs.

    python scripts/launch_multi_gpu.py --task 1                 # train -> evaluate/qualitative -> diagnostics -> aggregate
    python scripts/launch_multi_gpu.py --task 1 --stage train
    python scripts/launch_multi_gpu.py --task 2 --stage all     # prepare -> train(196) [+ train(49) on the 2nd GPU]
    python scripts/launch_multi_gpu.py --task 3                 # offline steps on CPU while live generation uses a GPU
    python scripts/launch_multi_gpu.py --task 1 --gpus 0,1 --dry_run
Every job logs to outputs/logs/<job>.log; the first failing job stops the whole run.
"""

import argparse
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def detect_gpus() -> List[int]:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20).stdout
        n = len([line for line in out.splitlines() if line.startswith("GPU ")])
        return list(range(n))
    except Exception:  # noqa: BLE001
        return []


def job(name: str, cmds: List[List[str]], deps: Optional[List[str]] = None, gpu: bool = True) -> Dict:
    return {"name": name, "cmds": cmds, "deps": deps or [], "gpu": gpu}


def t1(*args: str) -> List[str]:
    return [PY, os.path.join(ROOT, "scripts", "run_task1.py"), *args]


def t2(*args: str) -> List[str]:
    return [PY, os.path.join(ROOT, "scripts", "run_task2.py"), *args]


def t3(*args: str) -> List[str]:
    return [PY, os.path.join(ROOT, "scripts", "run_task3.py"), *args]


VARIANT_FOR = {
    "baseline": None, "full_ft_midpoint": "full_ft", "full_ft_final": "full_ft",
    "full_ft_l2_sp_final": "full_ft_l2_sp", "lora_4_final": "lora_4", "lora_16_final": "lora_16",
    "oplora_16_final": "oplora_16",
}


def jobs_task1(stage: str) -> List[Dict]:
    jobs: List[Dict] = []
    train_order = ["full_ft_l2_sp", "full_ft", "oplora_16", "lora_16", "lora_4"]  # heaviest first
    if stage in ("train", "all"):
        for v in train_order:
            jobs.append(job(f"train_{v}", [t1("--action", "train", "--variant", v)]))
    if stage in ("eval", "all"):
        for cond, variant in VARIANT_FOR.items():
            deps = [f"train_{variant}"] if (variant and stage == "all") else []
            jobs.append(job(f"eval_{cond}", [t1("--action", "evaluate", "--conditions", cond),
                                             t1("--action", "qualitative", "--conditions", cond)], deps))
        jobs.append(job("diagnostics", [t1("--action", "diagnostics")],
                        [f"train_{v}" for v in train_order] if stage == "all" else []))
        jobs.append(job("aggregate", [t1("--action", "aggregate")],
                        [f"eval_{c}" for c in VARIANT_FOR], gpu=False))
    return jobs


def jobs_task2(stage: str) -> List[Dict]:
    jobs: List[Dict] = []
    if stage in ("prepare", "all"):
        jobs.append(job("t2_prepare", [t2("--action", "prepare"), t2("--action", "cache_features")]))
    pre = ["t2_prepare"] if stage == "all" else []
    if stage in ("train", "all"):
        jobs.append(job("t2_train_196", [t2("--action", "train"), t2("--action", "evaluate")], pre))
        jobs.append(job("t2_train_49", [t2("--action", "compress")], pre))  # optional experiment, 2nd GPU
    return jobs


def jobs_task3(stage: str) -> List[Dict]:
    return [
        job("t3_live", [t3("--step", "live", "--dataset", "math")], gpu=True),
        job("t3_offline", [t3("--step", "opportunity"), t3("--step", "scoring"), t3("--step", "signals"),
                           t3("--step", "adaptive")], gpu=False),
    ]


def run_schedule(jobs: List[Dict], gpus: List[int], logdir: str, dry_run: bool) -> None:
    os.makedirs(logdir, exist_ok=True)
    if dry_run:
        for j in jobs:
            print(f"[{'GPU' if j['gpu'] else 'CPU'}] {j['name']}  deps={j['deps']}")
            for c in j["cmds"]:
                print("    ", " ".join(c))
        return

    pending, done = list(jobs), set()
    running: Dict[object, Dict] = {}   # slot -> {"job", "proc", "log"}
    cpu_slots = ["cpu0", "cpu1"]
    t0 = time.time()

    def start(j: Dict, slot) -> None:
        env = dict(os.environ)
        if isinstance(slot, int):
            env["CUDA_VISIBLE_DEVICES"] = str(slot)
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        shell_cmd = " && ".join(" ".join(f'"{a}"' if " " in a else a for a in c) for c in j["cmds"])
        log = open(os.path.join(logdir, f"{j['name']}.log"), "w", encoding="utf-8")
        proc = subprocess.Popen(shell_cmd, shell=True, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        running[slot] = {"job": j, "proc": proc, "log": log}
        print(f"[{time.time() - t0:7.0f}s] START {j['name']} on {'GPU ' + str(slot) if isinstance(slot, int) else 'CPU'}")

    while pending or running:
        for slot in list(gpus) + cpu_slots:
            if slot in running:
                continue
            want_gpu = isinstance(slot, int)
            if not want_gpu and not any(not j["gpu"] for j in pending):
                continue
            for j in pending:
                if j["gpu"] == want_gpu and set(j["deps"]) <= done:
                    pending.remove(j)
                    start(j, slot)
                    break
        for slot, rec in list(running.items()):
            rc = rec["proc"].poll()
            if rc is None:
                continue
            rec["log"].close()
            name = rec["job"]["name"]
            del running[slot]
            if rc != 0:
                for other in running.values():
                    other["proc"].terminate()
                raise SystemExit(f"Job {name} FAILED (exit {rc}); see {os.path.join(logdir, name + '.log')}")
            done.add(name)
            print(f"[{time.time() - t0:7.0f}s] DONE  {name}")
        if pending and not running:
            raise SystemExit(f"Deadlock: nothing runnable but pending={[j['name'] for j in pending]}")
        time.sleep(3)
    print(f"All jobs finished in {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--stage", default="all", help="task1: train|eval|all; task2: prepare|train|all")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU indices (default: auto-detect)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else detect_gpus()
    if not gpus and not args.dry_run:
        raise SystemExit("No GPU found (nvidia-smi). Use --gpus to override.")
    print(f"GPUs: {gpus}")
    jobs = {1: jobs_task1, 2: jobs_task2, 3: jobs_task3}[args.task](args.stage)
    run_schedule(jobs, gpus or [0], os.path.join(ROOT, "outputs", "logs"), args.dry_run)


if __name__ == "__main__":
    main()
