#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
uv_bin="$project_dir/.tools/uv"
python_dir="$project_dir/.local/share/uv/python"
uv_version="0.12.22"
uv_installer_sha256="58488ae8dbd0773134c92c85e901430e33f99d975bd7f929d26aa9ab0c2f9390"

if [ ! -x "$uv_bin" ]; then
    mkdir -p "$project_dir/.tools"
    installer=$(mktemp "$project_dir/.tools/uv-installer.XXXXXX")
    trap 'rm -f "$installer"' 0 1 2 15
    curl -LsSf "https://astral.sh/uv/$uv_version/install.sh" -o "$installer"
    if command -v sha256sum >/dev/null 2>&1; then
        actual_sha256=$(sha256sum "$installer" | awk '{print $1}')
    elif command -v shasum >/dev/null 2>&1; then
        actual_sha256=$(shasum -a 256 "$installer" | awk '{print $1}')
    else
        echo "A SHA-256 utility is required to verify the uv installer." >&2
        exit 1
    fi
    if [ "$actual_sha256" != "$uv_installer_sha256" ]; then
        echo "uv installer checksum mismatch; refusing to execute it." >&2
        exit 1
    fi
    env UV_INSTALL_DIR="$project_dir/.tools" UV_NO_MODIFY_PATH=1 sh "$installer"
    rm -f "$installer"
    trap - 0 1 2 15
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
