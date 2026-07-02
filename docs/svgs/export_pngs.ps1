<#
.SYNOPSIS
  Export the architecture diagrams (svgs/diagram_*.html) to transparent-background PNGs.

.DESCRIPTION
  Renders each diagram_*.html in this folder with headless Chrome (or Edge) and writes a PNG
  with a fully transparent page background, so the images drop straight onto slides/docs.

  The page background is forced transparent for the EXPORT ONLY -- a temp copy of each HTML is
  made with `background:#f8f8f6` swapped for `background:transparent`; the source files are not
  modified (they keep their light background when you open them in a browser).

  Window height is derived from each SVG's viewBox so the PNG is sized to the diagram. The
  20px body padding is kept as a small transparent margin.

.PARAMETER Scale
  Device pixel scale (default 2 = crisp for slides; use 3 for print, 1 for small files).

.PARAMETER OutDir
  Output folder (default: svgs\png).

.EXAMPLE
  .\export_pngs.ps1
.EXAMPLE
  .\export_pngs.ps1 -Scale 3
#>
[CmdletBinding()]
param(
    [double]$Scale = 2,
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $OutDir) { $OutDir = Join-Path $here "png" }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# --- locate a Chromium browser (Chrome preferred, then Edge) ---
$candidates = @(
    (Join-Path $env:ProgramFiles "Google\Chrome\Application\chrome.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Google\Chrome\Application\chrome.exe"),
    (Join-Path $env:ProgramFiles "Microsoft\Edge\Application\msedge.exe"),
    (Join-Path ${env:ProgramFiles(x86)} "Microsoft\Edge\Application\msedge.exe")
)
$browser = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $browser) { throw "No Chrome/Edge found. Install one, or edit `$candidates in this script." }
Write-Host "Browser : $browser"
Write-Host "Scale   : ${Scale}x"
Write-Host "Out dir : $OutDir`n"

$files = Get-ChildItem -Path $here -Filter "diagram_*.html" | Sort-Object Name
if (-not $files) { throw "No diagram_*.html found in $here" }

foreach ($f in $files) {
    # Read as UTF-8 (PS 5.1 Get-Content defaults to ANSI, which would mojibake — / · ≥ ₂ etc.)
    $html = Get-Content -Raw -Encoding UTF8 -LiteralPath $f.FullName

    # window height from the SVG viewBox ("0 0 W H"); svg renders ~860 wide in a 900px window
    $winW = 900
    $winH = 670
    if ($html -match 'viewBox="0 0 \d+ (\d+)"') { $winH = [int]$Matches[1] + 70 }

    # transparent-bg copy for export only (source untouched), written UTF-8 without BOM
    $tmpHtml = $html -replace 'background:\s*#f8f8f6', 'background: transparent'
    $tmp = Join-Path $env:TEMP ("export_" + $f.BaseName + ".html")
    [System.IO.File]::WriteAllText($tmp, $tmpHtml, (New-Object System.Text.UTF8Encoding $false))

    $out = Join-Path $OutDir ($f.BaseName + ".png")
    if (Test-Path $out) { Remove-Item $out -Force }
    $uri = "file:///" + ($tmp -replace '\\', '/')

    $chromeArgs = @(
        "--headless=new", "--disable-gpu", "--hide-scrollbars",
        "--default-background-color=00000000",          # transparent (ARGB, alpha=00)
        "--force-device-scale-factor=$Scale",
        "--window-size=$winW,$winH",
        "--screenshot=$out",
        $uri
    )

    # Start-Process avoids PS 5.1 wrapping native stderr as errors; redirect chatter to temp files.
    $log = [System.IO.Path]::GetTempFileName()
    Start-Process -FilePath $browser -ArgumentList $chromeArgs -NoNewWindow -Wait `
        -RedirectStandardError $log -RedirectStandardOutput "$log.out"
    Remove-Item $tmp, $log, "$log.out" -ErrorAction SilentlyContinue

    if (Test-Path $out) {
        $leaf = $f.BaseName + ".png"
        $pxw = [int]($winW * $Scale)
        $pxh = [int]($winH * $Scale)
        $kb = [math]::Round((Get-Item $out).Length / 1KB)
        Write-Host ("  {0,-30} -> png\{1,-26} {2}x{3}px  {4} KB" -f $f.Name, $leaf, $pxw, $pxh, $kb)
    }
    else {
        Write-Warning "  $($f.Name) -> FAILED (no PNG written)"
    }
}
Write-Host "`nDone. Transparent PNGs in: $OutDir"
