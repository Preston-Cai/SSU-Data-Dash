param(
    [Parameter(Mandatory = $true)][string]$PresentationPath,
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)

$ErrorActionPreference = 'Stop'
$msoTrue = -1
$app = New-Object -ComObject PowerPoint.Application
$app.Visible = $msoTrue
$presentation = $null
try {
    New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
    $presentation = $app.Presentations.Open((Resolve-Path $PresentationPath).Path, $msoTrue, 0, 0)
    $presentation.Export((Resolve-Path $OutputDirectory).Path, 'PNG', 1920, 1080)
}
finally {
    if ($presentation) { $presentation.Close() }
    $app.Quit()
    if ($presentation) { [System.Runtime.InteropServices.Marshal]::ReleaseComObject($presentation) | Out-Null }
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
