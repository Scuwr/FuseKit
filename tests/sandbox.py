import subprocess
import re
import math
import sys

def get_gpu_status():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            idx, name, total, used, free = re.split(r",\s*", line)
            gpus.append({
                "index": int(idx),
                "name": name,
                "total_MB": math.floor(int(total) / 1024) * 1000,
                "used_MB":  math.floor(int(used)  / 1024) * 1000,
                "free_MB":  math.floor(int(free)  / 1024) * 1000,
            })
        return gpus
    except Exception as e:
        print(f"Failed to query nvidia-smi: {e}", file=sys.stderr)
        return []

def get_available_gpus(threshold=0.9):
    gpus = get_gpu_status()
    qualifying = [g for g in gpus if g["free_MB"] >= threshold * g["total_MB"]]
    if not qualifying:
        return None
    memory_limit = min(qualifying, key=lambda g: g["free_MB"])["free_MB"]
    devices = [g["index"] for g in qualifying]
    return devices, memory_limit

if __name__ == "__main__":
    devices, memory_limit = get_available_gpus()    
    print(f"GPUs ≥90% free: {devices}")
    print(f"Smallest qualifying GPU free memory: {memory_limit} MB")