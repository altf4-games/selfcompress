# Extended batch: x1 (k=8) now, then x2 (new models, once converted), then x3 (controls, MATH extension, seed 3). Log: analysis\chain_ext.log
Set-Location (Split-Path -Parent $PSScriptRoot)
$log = "analysis\chain_ext.log"
"chain_ext started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $log
python scripts/queue.py configs/queue_x1.txt *>> $log
"x1 finished $(Get-Date -Format 'HH:mm:ss')" | Add-Content $log
while (-not (Select-String -Path analysis\convert_extra.log -Pattern '^convert_extra finished' -Quiet)) { Start-Sleep -Seconds 30 }
python scripts/queue.py configs/queue_x2.txt *>> $log
"x2 finished $(Get-Date -Format 'HH:mm:ss')" | Add-Content $log
python scripts/queue.py configs/queue_x3.txt *>> $log
"x3 finished $(Get-Date -Format 'HH:mm:ss')" | Add-Content $log
