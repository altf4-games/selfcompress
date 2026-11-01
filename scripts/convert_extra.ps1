# Wait for scripts/download_extra.py, then convert Qwen3-4B (-> Q4_K_M) and R1-Distill-Qwen-1.5B (-> F16). Log: analysis\convert_extra.log
Set-Location (Split-Path -Parent $PSScriptRoot)
$log = "analysis\convert_extra.log"
"convert_extra started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $log
while (-not (Select-String -Path analysis\download_extra.log -Pattern '^done R1-Distill' -Quiet)) { Start-Sleep -Seconds 30 }
$conv = (Get-ChildItem tools\dl\srcx -Recurse -Filter convert_hf_to_gguf.py | Select-Object -First 1).FullName
python $conv models\hf\R1-Distill-Qwen-1.5B --outtype f16 --outfile models\R1-Distill-Qwen-1.5B-F16.gguf *>> $log
python $conv models\hf\Qwen3-4B --outtype f16 --outfile models\Qwen3-4B-F16.gguf *>> $log
tools\llama\llama-quantize.exe models\Qwen3-4B-F16.gguf models\Qwen3-4B-Q4_K_M.gguf Q4_K_M *>> $log
"convert_extra finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Add-Content $log
