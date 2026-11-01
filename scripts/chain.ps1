# Wait for the running MATH queue, then run the final queue, then finalize (tables, paper, commit, push).
Set-Location (Split-Path -Parent $PSScriptRoot)
$log = "analysis\chain.log"
"chain started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $log
while (Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine -like '*queue.py*queue_math.txt*' }) {
    Start-Sleep -Seconds 60
}
"queue_math finished $(Get-Date -Format 'HH:mm:ss')" | Add-Content $log
python scripts/queue.py configs/queue_final.txt *>> $log
"queue_final finished $(Get-Date -Format 'HH:mm:ss')" | Add-Content $log
.\scripts\finalize.ps1
"chain finished $(Get-Date -Format 'HH:mm:ss')" | Add-Content $log
