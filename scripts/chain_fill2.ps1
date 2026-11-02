# After chain_ext.ps1: fill the MATH-500 extension generations that overflowed the context pool. Log: analysis\chain_fill2.log
Set-Location (Split-Path -Parent $PSScriptRoot)
$log = "analysis\chain_fill2.log"
"fill2 started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $log
while (-not (Select-String -Path analysis\chain_ext.log -Pattern 'x3 finished' -Quiet)) { Start-Sleep -Seconds 20 }
python scripts/queue.py configs/queue_fill2.txt *>> $log
"fill2 finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Add-Content $log
