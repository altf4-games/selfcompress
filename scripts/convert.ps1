# Usage: .\scripts\convert.ps1 Qwen3-1.7B  -> models\<name>-F16.gguf, -Q8_0.gguf, -Q4_K_M.gguf (same pipeline for all precisions)
param([string]$Name)
$conv = (Get-ChildItem tools\dl\srcx -Recurse -Filter convert_hf_to_gguf.py | Select-Object -First 1).FullName
python $conv models\hf\$Name --outtype f16 --outfile models\$Name-F16.gguf
foreach ($q in "Q8_0","Q4_K_M") { tools\llama\llama-quantize.exe models\$Name-F16.gguf models\$Name-$q.gguf $q }
