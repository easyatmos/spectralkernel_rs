#!/bin/sh

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
PYTHON_PATH=$(which python)
TEST_TARGET=${TEST_TARGET:-"tests/test_sphere_ops.py"}

cd "$ROOT_DIR" || exit 1

# 检查是否找到 Python
# Check if Python is found
if [ -z "$PYTHON_PATH" ]; then
    echo "Error: Python not found in PATH."
    exit 1
fi

# 获取 Python 主次版本号（例如 "3.13"）
# Get Python primary and secondary version numbers (e.g. "3.13")
PYTHON_VERSION=$("$PYTHON_PATH" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# 设置环境变量，强制 pre-commit 使用当前 Python
# Set environment variables to force pre-commit to use the current Python
export PRE_COMMIT_USE_SYSTEM_PYTHON=1
export VIRTUALENV_PYTHON="$PYTHON_PATH"

# 运行 pytest
# Run pytest
echo "Using Python: $PYTHON_PATH (Version: $PYTHON_VERSION)"
"$PYTHON_PATH" -m pytest -q "$TEST_TARGET"
