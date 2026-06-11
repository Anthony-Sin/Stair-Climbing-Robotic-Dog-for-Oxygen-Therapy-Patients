# 2>nul & cls & powershell -NoProfile -ExecutionPolicy Bypass -Command "IEX ([System.IO.File]::ReadAllText('%~f0'))" & exit /b

Write-Host "===================================================================="
Write-Host "          AUTOMATED PROJECT CODE AND LOG SUMMARY GENERATOR"
Write-Host "===================================================================="

$ProjectRoot = Get-Location
$LogFolder   = Join-Path $ProjectRoot "log"

# Output targets set explicitly to your project root
$OutputRoot  = Join-Path $ProjectRoot "fullcode_root.txt"
$OutputLog   = Join-Path $ProjectRoot "fullcode_log.txt"

# --------------------------------------------------------------------
# STEP 1: Compile Project Source Code (Using your exact filters + leak patch)
# --------------------------------------------------------------------
Write-Host "-> Compiling project source code into fullcode_root.txt..."

Get-ChildItem -Path $ProjectRoot -Recurse -File | Where-Object {  
    # 1. Fixed the \b leak to capture build_isolated, install_isolated, etc.
    ($_.FullName -notmatch '\\(build|install|log|__pycache__|\.git|\.venv|doc|resource|config|ros2_ws|tests|node_modules|dist)(_|-|\b)') -and
    # 2. Exclude non-code extensions
    ($_.Extension -notmatch '\.(md|pdf|json|yaml|xml|txt|svg|gif|dat|cfg|lock|css)$') -and
    # 3. Exclude specific configuration / meta files
    ($_.Name -notmatch '^(LICENSE|README|VERSION|change.*|eslint\.config\.js|vite\.config\.ts|tsconfig.*|__init__.py)$') -and
    # 4. Target your precise development/source files
    ($_.Extension -match '\.(cpp|hpp|h|cu|cuh|c|cc|py|vert|frag|comp|glsl|sh|tsx|ts)$' -or $_.Name -like 'Dockerfile*')
} | ForEach-Object {  
    "`n`n====================================================================`nFILE: $($_.FullName)`n===================================================================="  
    Get-Content $_.FullName -ErrorAction SilentlyContinue
} | Out-File -Encoding utf8 $OutputRoot


# --------------------------------------------------------------------
# STEP 2: Compile Log Folder Files Only
# --------------------------------------------------------------------
if (Test-Path $LogFolder) {
    Write-Host "-> Compiling log folder files into fullcode_log.txt..."
    Get-ChildItem -Path $LogFolder -Recurse -File | Where-Object { 
        ($_.Name -notmatch 'fullcode_(root|log)\.txt') -and 
        ($_.FullName -notmatch '\\(\.git|\.venv|node_modules)\b') -and 
        ($_.Extension -match '\.(log|txt|jsonl)$') 
    } | ForEach-Object { 
        "`n`n====================================================================`nFILE: $($_.FullName)`n===================================================================="
        Get-Content $_.FullName -ErrorAction SilentlyContinue 
    } | Out-File -Encoding utf8 $OutputLog
}

Write-Host ""
Write-Host "[SUCCESS] Operations complete! Logs and Code separated. Exiting..." -ForegroundColor Green
Start-Sleep -Seconds 1