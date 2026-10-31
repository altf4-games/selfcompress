"""Run configs/queue.txt in order. Restarts llama-server only when (model, precision) changes.

Usage: python scripts/queue.py            (resumable: finished units are skipped; each config gets 2 passes)
Only stops the server process it started itself.
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAMES = {"q17": "Qwen3-1.7B", "q06": "Qwen3-0.6B"}
PREC = {"f16": "F16", "q8": "Q8_0", "q4": "Q4_K_M"}


def healthy():
    try:
        return urllib.request.urlopen("http://127.0.0.1:8089/health", timeout=2).status == 200
    except Exception:
        return False


def start(m, p, slots, ctx):
    gguf = ROOT / "models" / f"{NAMES[m]}-{PREC[p]}.gguf"
    log = open(ROOT / "runs" / f"server_{m}_{p}.log", "a")
    proc = subprocess.Popen([str(ROOT / "tools/llama/llama-server.exe"), "-m", str(gguf), "--jinja",
                             "--reasoning-format", "none", "-ngl", "99", "-np", str(slots), "-c", str(ctx),
                             "--kv-unified", "--port", "8089", "--host", "127.0.0.1"], stdout=log, stderr=log)
    for _ in range(120):
        if healthy():
            return proc
        time.sleep(2)
    proc.terminate()
    raise RuntimeError("server did not come up")


def main():
    qfile = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "configs" / "queue.txt"
    rows = [l.split() for l in qfile.read_text().splitlines() if l.strip()]
    cur, proc = None, None
    try:
        for m, p, slots, ctx, rid in rows:
            if cur != (m, p, slots):
                if proc:
                    proc.terminate(); proc.wait()
                assert not healthy(), "something else is already serving on 8089"
                proc, cur = start(m, p, slots, ctx), (m, p, slots)
            for _ in range(2):  # second pass fills any units that errored in the first
                subprocess.run([sys.executable, str(ROOT / "src/run.py"), str(ROOT / "configs" / f"{rid}.yaml")])
    finally:
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
