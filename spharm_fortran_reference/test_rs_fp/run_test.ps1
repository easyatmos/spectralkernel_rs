Get-ChildItem -Filter "test_*.py" | ForEach-Object {
    $logFile = "$($_.BaseName).log"
    Write-Host "Running $($_.Name)..."
    python $_.Name > $logFile 2>&1
}

# 合并时在每个文件之间添加分隔线
Get-ChildItem *.log | ForEach-Object {
    "===== $($_.Name) ====="
    Get-Content $_
} | Set-Content merged.log