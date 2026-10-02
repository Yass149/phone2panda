#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
uv_bin="$project_dir/.tools/uv"
python_dir="$project_dir/.local/share/uv/python"
uv_version="0.12.22"

if [ ! -x "$uv_bin" ]; then
  mkdir -p "$project_dir/.tools"
  curl -LsSf "https://astral.sh/uv/$uv_version/install.sh" | \
    env UV_INSTALL_DIR="$project_dir/.tools" UV_NO_MODIFY_PATH=1 sh
fi

export UV_CACHE_DIR="$project_dir/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$python_dir"

if ! "$uv_bin" python find 3.11.17 >/dev/null 2>&1; then
  "$uv_bin" python install 3.11.17
fi
python_bin=$("$uv_bin" python find 3.11.17)

if [ ! -x "$project_dir/.venv/bin/python" ]; then
  "$uv_bin" venv --python "$python_bin" "$project_dir/.venv"
fi

"$uv_bin" sync --python "$project_dir/.venv/bin/python" --frozen
