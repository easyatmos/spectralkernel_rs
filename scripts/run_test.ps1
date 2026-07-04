$ErrorActionPreference = "Stop"

$RootDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$PYTHON_PATH = (Get-Command python).Source
$TEST_TARGET = if ($env:TEST_TARGET) {
    $env:TEST_TARGET
} else {
    "tests\test_sphere_ops.py"
}

Set-Location $RootDir

# 检查是否找到 Python
# Check if Python is found
if (-not $PYTHON_PATH) {
    Write-Host "Error: Python not found in PATH."
    exit 1
}

# 获取 Python 主次版本号（例如 "3.13"）
# Get Python primary and secondary version numbers (e.g. "3.13")
$PYTHON_VERSION = & $PYTHON_PATH -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"

# 设置环境变量，强制 pre-commit 使用当前 Python
# Set environment variables to force pre-commit to use the current Python
$env:PRE_COMMIT_USE_SYSTEM_PYTHON = "1"
$env:VIRTUALENV_PYTHON = "$PYTHON_PATH"

# 运行 pytest
# Run pytest
Write-Host "Using Python: $PYTHON_PATH (Version: $PYTHON_VERSION)"
& $PYTHON_PATH -m pytest -q $TEST_TARGET
