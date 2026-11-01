# Regenerate tables and figures, rebuild the paper, commit and push. Log: analysis\finalize.log
# Usage: .\scripts\finalize.ps1
Set-Location (Split-Path -Parent $PSScriptRoot)
$log = "analysis\finalize.log"
"finalize started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $log
python src/make_tables.py *>> $log
python src/make_math_table.py *>> $log
Push-Location paper
latexmk -pdf -interaction=nonstopmode main.tex *>> ..\$log
latexmk -pdf -interaction=nonstopmode main.tex *>> ..\$log
Pop-Location
git add -A *>> $log
git commit -m "Final run batch: seed 2 for 0.6B quantized, MATH-500 cells; regenerated tables, figures and paper" *>> $log
git push origin main *>> $log
"finalize finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Add-Content $log
