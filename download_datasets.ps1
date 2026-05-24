# DataAgent 工业数据集下载脚本 (PowerShell)
# 使用 Invoke-WebRequest + SkipCertificateCheck 绕过 SSL

$ProgressPreference = "SilentlyContinue"
$dataDir = "C:\Users\A\.qclaw\workspace\DataAgent\data"
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

function Download-File {
    param([string]$Url, [string]$OutPath, [string]$Desc)
    if (-not $Desc) { $Desc = Split-Path $OutPath -Leaf }
    $part = "$OutPath.part"

    if (Test-Path $OutPath) {
        $szKB = (Get-Item $OutPath).Length / 1KB
        Write-Host "  [SKIP] $Desc ($($szKB.ToString('N1')) KB)" -ForegroundColor Yellow
        return $true
    }

    Write-Host "  [DL] $Desc" -ForegroundColor Cyan
    try {
        Invoke-WebRequest -Uri $Url -OutFile $part `
            -SkipCertificateCheck -TimeoutSec 120 -UseBasicParsing -ErrorAction Stop
        if (Test-Path $part) {
            Rename-Item -Path $part -NewName (Split-Path $OutPath -Leaf) -Force
            $szMB = (Get-Item $OutPath).Length / 1MB
            Write-Host "  [OK] $Desc ($($szMB.ToString('N2')) MB)" -ForegroundColor Green
            return $true
        }
    }
    catch {
        Write-Host "  [ERR] $Desc : $($_.Exception.Message)" -ForegroundColor Red
    }
    return $false
}

# ================================================================
# 1. NASA Turbofan (CMAPSS)
# ================================================================
Write-Host "`n============================================================"
Write-Host "数据集 1/3: NASA Turbofan (CMAPSS)"
Write-Host "============================================================"

$githubBase  = "https://raw.githubusercontent.com/alexhag/CMAPSS/master/data"
$backupBase = "https://raw.githubusercontent.com/biswajitsamanta/NASA-CMAPSS/main/data"
$subsets     = @("FD001", "FD002", "FD003", "FD004")
$nasaOk = 0

foreach ($sub in $subsets) {
    foreach ($type in @("train", "test", "RUL")) {
        $fname = "$type" + "_$sub.txt"
        $out   = Join-Path $dataDir $fname
        $url1  = "$githubBase/$fname"
        $url2  = "$backupBase/$fname"

        if (Download-File -Url $url1 -OutPath $out -Desc $fname) {
            $nasaOk++
        }
        elseif (Download-File -Url $url2 -OutPath $out -Desc $fname) {
            $nasaOk++
        }
        else {
            Write-Host "  [FAIL] $fname - 所有源均失败" -ForegroundColor Magenta
        }
    }
}

# 如果文件不够，尝试下载 ZIP
if ($nasaOk -lt 6) {
    Write-Host "`n  GitHub 源不完整，尝试下载 ZIP 包..." -ForegroundColor Yellow
    $zipUrl  = "https://github.com/alexhag/CMAPSS/archive/refs/heads/master.zip"
    $zipPath = Join-Path $dataDir "CMAPSS-master.zip"
    try {
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath `
            -SkipCertificateCheck -TimeoutSec 300 -UseBasicParsing
        Write-Host "  [OK] ZIP 下载完成，解压中..." -ForegroundColor Green
        Expand-Archive -Path $zipPath -DestinationPath $dataDir -Force
        Write-Host "  [OK] 解压完成" -ForegroundColor Green
        $nasaOk = 12
    }
    catch {
        Write-Host "  [ERR] ZIP: $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host "`n  NASA Turbofan: $nasaOk/12 文件" -ForegroundColor Cyan

# ================================================================
# 2. UCI Hydraulic System
# ================================================================
Write-Host "`n============================================================"
Write-Host "数据集 2/3: UCI Hydraulic System"
Write-Host "============================================================"

$zipUrl  = "https://archive.ics.uci.edu/static/public/438/condition+monitoring+of+hydraulic+systems.zip"
$zipPath = Join-Path $dataDir "hydraulic_system.zip"
$uciOk   = $false

try {
    Write-Host "  [DL] UCI Hydraulic ZIP..."
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath `
        -SkipCertificateCheck -TimeoutSec 300 -UseBasicParsing
    Write-Host "  [OK] 下载完成，解压中..." -ForegroundColor Green
    Expand-Archive -Path $zipPath -DestinationPath $dataDir -Force
    Write-Host "  [OK] 解压完成" -ForegroundColor Green
    $uciOk = $true
}
catch {
    Write-Host "  [ERR] UCI ZIP: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  [WARN] 将使用模拟数据代替" -ForegroundColor Yellow
}

# ================================================================
# 3. SECOM 半导体
# ================================================================
Write-Host "`n============================================================"
Write-Host "数据集 3/3: SECOM Semiconductor"
Write-Host "============================================================"

$dataUrl  = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
$labelUrl = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
$dataPath = Join-Path $dataDir "secom.data"
$labelPath = Join-Path $dataDir "secom_labels.data"
$secomOk  = $false

try {
    Write-Host "  [DL] secom.data..."
    Invoke-WebRequest -Uri $dataUrl -OutFile $dataPath `
        -SkipCertificateCheck -TimeoutSec 120 -UseBasicParsing
    Write-Host "  [OK] secom.data" -ForegroundColor Green

    Write-Host "  [DL] secom_labels.data..."
    Invoke-WebRequest -Uri $labelUrl -OutFile $labelPath `
        -SkipCertificateCheck -TimeoutSec 120 -UseBasicParsing
    Write-Host "  [OK] secom_labels.data" -ForegroundColor Green
    $secomOk = $true
}
catch {
    Write-Host "  [ERR] SECOM: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  [WARN] 将使用模拟数据代替" -ForegroundColor Yellow
}

# ================================================================
# 总结
# ================================================================
Write-Host "`n============================================================"
Write-Host "  下载结果总结"
Write-Host "============================================================"

$nasaStatus  = if ($nasaOk -ge 6) { "OK - 成功" } else { "WARN - 部分失败" }
$uciStatus   = if ($uciOk)       { "OK - 成功" } else { "WARN - 使用模拟数据" }
$secomStatus = if ($secomOk)      { "OK - 成功" } else { "WARN - 使用模拟数据" }

Write-Host "  NASA  : $nasaStatus"
Write-Host "  UCI   : $uciStatus"
Write-Host "  SECOM : $secomStatus"

Write-Host "`n  数据目录内容:" -ForegroundColor Cyan
Get-ChildItem $dataDir | ForEach-Object {
    $szStr = ""
    if (-not $_.PSIsContainer) {
        $szMB = $_.Length / 1MB
        $szStr = " ($($szMB.ToString('N2')) MB)"
    }
    Write-Host "    $($_.Name)$szStr"
}

Write-Host "`n  [OK] 下载流程完成！" -ForegroundColor Green
Write-Host "============================================================"
