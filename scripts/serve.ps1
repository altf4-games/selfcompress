# Usage: .\scripts\serve.ps1 models\Qwen3-1.7B-F16.gguf [slots] [total_ctx]
# --reasoning-format none keeps <think>...</think> in the raw output so we can split it ourselves.
# total_ctx is the shared KV pool across slots (8 GB VRAM cannot hold slots * 9216 for F16 1.7B).
param([string]$Gguf, [int]$Slots = 4, [int]$Ctx = 20480)
tools\llama\llama-server.exe -m $Gguf --jinja --reasoning-format none -ngl 99 -np $Slots -c $Ctx --kv-unified --port 8089 --host 127.0.0.1
