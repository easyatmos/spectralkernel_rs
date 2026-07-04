param(
    [string]$OutputDir = "target/rustdoc"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "Cargo.toml")) {
    throw "Please run this script in the root directory of the project that contains Cargo.toml."
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$env:RUSTDOCFLAGS = "--cfg docsrs"

cargo doc --no-deps --document-private-items --target-dir $OutputDir

Write-Host "rustdoc has been generated to $OutputDir/doc"
