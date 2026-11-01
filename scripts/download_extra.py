"""Download the extra HF checkpoints (Qwen3-4B, DeepSeek-R1-Distill-Qwen-1.5B) into models/hf/. Usage: python scripts/download_extra.py"""
import sys
from pathlib import Path

sys.path = [x for x in sys.path if Path(x or '.').resolve() != Path(__file__).resolve().parent]  # scripts/queue.py shadows stdlib queue
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
for repo, name in (("Qwen/Qwen3-4B", "Qwen3-4B"), ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "R1-Distill-Qwen-1.5B")):
    snapshot_download(repo, local_dir=ROOT / "models" / "hf" / name, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    print("done", name, flush=True)
