#!/usr/bin/env python3
import argparse
import subprocess


def query_gpus():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,pci.bus_id,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "bus_id": parts[1],
                "memory_used_mib": int(parts[2]),
                "utilization_gpu_percent": int(parts[3]),
            }
        )
    return rows


def query_busy_bus_ids():
    cmd = [
        "nvidia-smi",
        "--query-compute-apps=gpu_bus_id,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError:
        return set()

    busy = set()
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", maxsplit=3)]
        if len(parts) >= 2 and parts[1]:
            busy.add(parts[0])
    return busy


def main():
    parser = argparse.ArgumentParser(description="Select a free GPU for shared-server inference.")
    parser.add_argument("--max-memory-mib", type=int, default=500)
    parser.add_argument("--max-utilization", type=int, default=5)
    args = parser.parse_args()

    busy_bus_ids = query_busy_bus_ids()
    free = [
        row
        for row in query_gpus()
        if row["bus_id"] not in busy_bus_ids
        and row["memory_used_mib"] <= args.max_memory_mib
        and row["utilization_gpu_percent"] <= args.max_utilization
    ]
    if not free:
        raise SystemExit(
            "No free GPU found. Check nvidia-smi and set GPU_ID manually only after confirming an idle GPU."
        )
    print(free[0]["index"])


if __name__ == "__main__":
    main()
