param(
    [ValidateRange(1, 500)]
    [int]$Limit = 5
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path $PSScriptRoot -Parent
$sourceRoot = Join-Path $repoRoot "examples\musique"
$datasetRoot = Join-Path $repoRoot "workspace\musique"

if (-not $env:OPENAI_API_KEY) {
    throw "Set OPENAI_API_KEY before running the example."
}
if (-not $env:GRAPHRAG_API_KEY) {
    $env:GRAPHRAG_API_KEY = $env:OPENAI_API_KEY
}

if (-not (Test-Path $datasetRoot)) {
    New-Item -ItemType Directory -Path $datasetRoot | Out-Null
    Copy-Item -LiteralPath (Join-Path $sourceRoot "input") -Destination $datasetRoot -Recurse
    Copy-Item -LiteralPath (Join-Path $sourceRoot "qa-pairs") -Destination $datasetRoot -Recurse
}

Push-Location $repoRoot
try {
    if (-not (Test-Path (Join-Path $datasetRoot "settings.yaml"))) {
        python -m graphrag init --root $datasetRoot
    }
    python -m graphrag index --root $datasetRoot
    python -m esg_rag.preflight $datasetRoot --require-index
    python -m esg_rag.retrieve $datasetRoot $Limit
    python -m esg_rag.generate $datasetRoot rf-dqd
    python -m esg_rag.select $datasetRoot rf-dqd standard --limit $Limit
} finally {
    Pop-Location
}

