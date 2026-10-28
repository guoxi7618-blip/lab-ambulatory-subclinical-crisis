# Export each single-slide figure deck through PowerPoint: PDF (vector shapes) and 600-dpi PNG.
# Run: powershell -ExecutionPolicy Bypass -File export_figures_v4.ps1
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$slides = Join-Path $root 'Figures\v4\slides'
$out = Join-Path $root 'Figures\v4\export'
New-Item -ItemType Directory -Force $out | Out-Null
$app = New-Object -ComObject PowerPoint.Application
try {
    Get-ChildItem $slides -Filter '*.pptx' | ForEach-Object {
        $name = $_.BaseName
        $pres = $app.Presentations.Open($_.FullName, $true, $false, $false)
        $wPt = $pres.PageSetup.SlideWidth; $hPt = $pres.PageSetup.SlideHeight
        $wPx = [int][math]::Round($wPt / 72 * 600); $hPx = [int][math]::Round($hPt / 72 * 600)
        $pres.SaveAs((Join-Path $out "$name.pdf"), 32)
        $pres.Slides.Item(1).Export((Join-Path $out "$name.png"), 'PNG', $wPx, $hPx)
        $pres.Close()
        Write-Output "$name exported ($wPx x $hPx px)"
    }
} finally {
    $app.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
