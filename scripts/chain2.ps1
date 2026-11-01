# After chain.ps1 ends: fill the 4 MATH-500 k=4 prompts that failed with context overflow, then finalize again.
Set-Location (Split-Path -Parent $PSScriptRoot)
$log = "analysis\chain2.log"
"chain2 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $log
while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*scripts\chain.ps1*' }) { Start-Sleep -Seconds 60 }
python scripts/queue.py configs/queue_fill.txt *>> $log
.\scripts\finalize.ps1
"chain2 finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Add-Content $log
