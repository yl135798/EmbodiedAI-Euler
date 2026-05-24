# ============================================================
# DataAgent 数据集手动下载助手
# 运行方式: 右键 → 以 PowerShell 运行
# ============================================================

$ErrorActionPreference = "SilentlyContinue"
$DataDir = "$env:USERPROFILE\.qclaw\workspace\DataAgent\data"
New-Item -ItemType Directory -Path $DataDir -Force | Out-Null

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  DataAgent 数据集下载助手" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# ============================================================
# 1. NASA Turbofan (CMAPSS)
# ============================================================
Write-Host "`n[1/3] NASA Turbofan (CMAPSS) 数据集" -ForegroundColor Yellow
Write-Host "  官方来源: https://data.nasa.gov/dataset/CMAPSS-Data-Set/xxxx"
Write-Host "  Kaggle镜像: https://www.kaggle.com/datasets/behradmjn/nasa-cmapss"
Write-Host ""
Write-Host "  由于需要登录，请手动下载后放入:" -ForegroundColor Gray
Write-Host "  $DataDir" -ForegroundColor DarkGray
Write-Host ""
Read-Host "  按 Enter 打开 Kaggle 下载页面（需登录 Kaggle 账号）"
Start-Process "https://www.kaggle.com/datasets/behradmjn/nasa-cmapss"

Write-Host "  下载后，将以下文件放入 $DataDir :" -ForegroundColor Gray
Write-Host "    train_FD001.txt, test_FD001.txt, RUL_FD001.txt" -ForegroundColor DarkCyan
Write-Host "    train_FD002.txt, test_FD002.txt, RUL_FD002.txt" -ForegroundColor DarkCyan
Write-Host "    train_FD003.txt, test_FD003.txt, RUL_FD003.txt" -ForegroundColor DarkCyan
Write-Host "    train_FD004.txt, test_FD004.txt, RUL_FD004.txt" -ForegroundColor DarkCyan

# ============================================================
# 2. UCI Hydraulic System
# ============================================================
Write-Host "`n[2/3] UCI Hydraulic System 数据集 (230 MB)" -ForegroundColor Yellow
$uciUrl = "https://archive.ics.uci.edu/dataset/438"
Write-Host "  UCI 官方页面: $uciUrl" -ForegroundColor Gray
Read-Host "  按 Enter 打开 UCI 下载页面"
Start-Process $uciUrl

Write-Host "  下载后，解压并将以下文件放入 $DataDir :" -ForegroundColor Gray
Write-Host "    Cooler_Condition.csv" -ForegroundColor DarkCyan
Write-Host "    Valve_Condition.csv" -ForegroundColor DarkCyan
Write-Host "    Pump_Leak.csv" -ForegroundColor DarkCyan
Write-Host "    Accumulator_Condition.csv" -ForegroundColor DarkCyan
Write-Host "    System_Failure.csv" -ForegroundColor DarkCyan

# ============================================================
# 3. 检查已下载的文件
# ============================================================
Write-Host "`n[3/3] 检查已下载的文件..." -ForegroundColor Yellow
$files = Get-ChildItem $DataDir | Where-Object { !$_.PSIsContainer -and $_.Name -notlike "_*" }
if ($files.Count -eq 0) {
    Write-Host "  [EMPTY] 数据目录为空，请先下载数据集" -ForegroundColor Red
} else {
    Write-Host "  已找到 $($files.Count) 个文件:" -ForegroundColor Green
    foreach ($f in $files) {
        $sz = if ($f.Length -gt 1MB) { "{0:N1} MB" -f ($f.Length/1MB) } else { "{0:N1} KB" -f ($f.Length/1KB) }
        Write-Host "    $($f.Name) ($sz)" -ForegroundColor Cyan
    }
}

# ============================================================
# 4. 验证 SECOM 数据
# ============================================================
Write-Host "`n[验证] SECOM 数据..." -ForegroundColor Yellow
$secomData = Join-Path $DataDir "secom.data"
$secomLabel = Join-Path $DataDir "secom_labels.data"
if ((Test-Path $secomData) -and (Test-Path $secomLabel)) {
    $sz1 = (Get-Item $secomData).Length / 1MB
    $sz2 = (Get-Item $secomLabel).Length / 1KB
    Write-Host "  [OK] SECOM 数据完整 ($($sz1:N1) MB + $($sz2:N1) KB)" -ForegroundColor Green
} else {
    Write-Host "  [WARN] SECOM 数据不完整，正在用 curl 下载..." -ForegroundColor Yellow
    $urls = @(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
    )
    foreach ($url in $urls) {
        $fname = Split-Path $url -Leaf
        $out = Join-Path $DataDir $fname
        Write-Host "    下载: $fname ..."
        curl.exe -L -k -o $out $url
    }
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  下载完成后，运行以下命令测试:" -ForegroundColor Cyan
Write-Host "  python $env:USERPROFILE\.qclaw\workspace\DataAgent\demo.py" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Cyan

Read-Host "`n按 Enter 退出"
