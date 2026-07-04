$ErrorActionPreference = "Stop"

<#
Run the FLUX.2 img2img refine batch twice, in series:

1. normal images:
   F:\sample_images\test_run_02\normal
   denoise 0.45

2. mosaic images:
   F:\sample_images\test_run_02\mosaic
   denoise 0.65

Both runs use repeat-count 1. ComfyUI must already be running at the Python
runner's default server, http://127.0.0.1:8188.

Usage from the ComfyUI repo root:

    .\scripts\workflows\run_img2img_refine.ps1

Why staging is used:
ComfyUI's core SaveImage node saves under the ComfyUI output directory. This
wrapper stages outputs under ./output/_test_run_02_img2img/<label>, then moves
the generated PNGs into F:\sample_images\test_run_02\outputs.

To change folders or denoise values, edit the Invoke-Img2ImgRefineRun calls at
the bottom of this file.
#>

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $PSScriptRoot "run_img2img_refine.py"
$ComfyOutputRoot = Join-Path $RepoRoot "output"

$FinalOutputDir = "F:\sample_images\test_run_02\outputs"
$StageRoot = "_test_run_02_img2img"

New-Item -ItemType Directory -Force -Path $FinalOutputDir | Out-Null

function Invoke-Img2ImgRefineRun {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Label,

        [Parameter(Mandatory = $true)]
        [string] $InputDir,

        [Parameter(Mandatory = $true)]
        [double] $Denoise
    )

    $StageSubfolder = "$StageRoot/$Label"
    $StageDir = Join-Path $ComfyOutputRoot ($StageSubfolder -replace "/", [System.IO.Path]::DirectorySeparatorChar)
    $ResolvedOutputRoot = [System.IO.Path]::GetFullPath($ComfyOutputRoot)
    $ResolvedStageDir = [System.IO.Path]::GetFullPath($StageDir)

    if (-not $ResolvedStageDir.StartsWith($ResolvedOutputRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove staging directory outside ComfyUI output: $ResolvedStageDir"
    }

    if (Test-Path $StageDir) {
        Remove-Item -LiteralPath $StageDir -Recurse -Force
    }

    Write-Host "Running $Label img2img refine..."
    & $Python $Runner `
        --input-dir $InputDir `
        --denoise $Denoise `
        --repeat-count 1 `
        --output-dir $StageSubfolder

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    if (-not (Test-Path $StageDir)) {
        throw "Expected ComfyUI staged output directory was not created: $StageDir"
    }

    $Generated = Get-ChildItem -Path $StageDir -Filter "*.png" -File
    if (-not $Generated) {
        throw "No PNG outputs were found in staged output directory: $StageDir"
    }

    $Generated | Move-Item -Destination $FinalOutputDir -Force
    Write-Host "Moved $($Generated.Count) $Label output image(s) to $FinalOutputDir"
}

Invoke-Img2ImgRefineRun `
    -Label "normal" `
    -InputDir "F:\sample_images\test_run_02\normal" `
    -Denoise 0.45

Invoke-Img2ImgRefineRun `
    -Label "mosaic" `
    -InputDir "F:\sample_images\test_run_02\mosaic" `
    -Denoise 0.65
