param(
    [string]$FoundationSha = "",
    [string]$Remote = "origin",
    [string]$Root = "C:\Users\willj\.codex\worktrees"
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

git fetch $Remote

$foundationPath = Join-Path $Root "usm-foundation"
git worktree add $foundationPath -b codex/foundation-contracts-gates "$Remote/main"

if ([string]::IsNullOrWhiteSpace($FoundationSha)) {
    Write-Host "Foundation worktree created at $foundationPath"
    Write-Host "Commit foundation branch, then rerun with -FoundationSha <commit>."
    exit 0
}

$targets = @(
    @{ Path = "usm-p1"; Branch = "codex/p1-multi-source-ingestion" },
    @{ Path = "usm-p3"; Branch = "codex/p3-retrieval-stack-v2" },
    @{ Path = "usm-p6"; Branch = "codex/p6-multi-judge-panel" },
    @{ Path = "usm-p7"; Branch = "codex/p7-failure-taxonomy" },
    @{ Path = "usm-p9"; Branch = "codex/p9-robustness-suite" },
    @{ Path = "usm-p10"; Branch = "codex/p10-domain-transfer" },
    @{ Path = "usm-reco"; Branch = "codex/reco-interfaces-tests-ci-gates" }
)

foreach ($target in $targets) {
    $path = Join-Path $Root $target.Path
    git worktree add $path -b $target.Branch $FoundationSha
}

$integrationPath = Join-Path $Root "usm-integration"
git worktree add $integrationPath -b codex/integration-research-upgrades "$Remote/main"

Write-Host "Created program and integration worktrees from foundation SHA $FoundationSha"
