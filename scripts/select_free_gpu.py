import argparse
import subprocess
import sys


def query_gpus():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    rows = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "memory_free_mib": int(parts[1]),
                "memory_total_mib": int(parts[2]),
                "utilization_gpu_percent": int(parts[3]),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Select a GPU with the most free memory for shared-server inference.")
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=12000,
        help="이 값 이상의 여유 메모리를 가진 GPU만 후보로 삼는다(기본 12GB; INT8/INT4 적재 가능 수준).",
    )
    parser.add_argument(
        "--max-utilization",
        type=int,
        default=80,
        help="이 점유율(%)을 초과하는 GPU는 제외한다(기본 80).",
    )
    args = parser.parse_args()

    gpus = query_gpus()
    if not gpus:
        raise SystemExit("nvidia-smi에서 GPU 정보를 가져오지 못했습니다.")

    candidates = [
        g
        for g in gpus
        if g["memory_free_mib"] >= args.min_free_mib and g["utilization_gpu_percent"] <= args.max_utilization
    ]
    if not candidates:
        best = max(gpus, key=lambda g: g["memory_free_mib"])
        raise SystemExit(
            f"조건을 만족하는 GPU가 없습니다(최소 여유 {args.min_free_mib}MiB, 점유율 {args.max_utilization}% 이하).\n"
            f"가장 여유 있는 GPU는 index {best['index']} (여유 {best['memory_free_mib']}MiB, "
            f"점유율 {best['utilization_gpu_percent']}%)입니다. "
            f"필요하면 GPU_ID로 직접 지정하거나 --min-free-mib 값을 낮추세요."
        )

    # 여유 메모리 최대 → 동률이면 점유율 최소
    chosen = sorted(candidates, key=lambda g: (-g["memory_free_mib"], g["utilization_gpu_percent"]))[0]
    print(
        f"[select_free_gpu] GPU {chosen['index']} 선택 "
        f"(여유 {chosen['memory_free_mib']}/{chosen['memory_total_mib']}MiB, "
        f"점유율 {chosen['utilization_gpu_percent']}%)",
        file=sys.stderr,
    )
    print(chosen["index"])


if __name__ == "__main__":
    main()
