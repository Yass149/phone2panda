from __future__ import annotations

import re
from pathlib import Path


def test_uv_bootstrap_verifies_the_pinned_installer_before_execution() -> None:
    script = Path("tools/bootstrap.sh").read_text(encoding="utf-8")

    assert "uv_version=\"0.12.22\"" in script
    assert re.search(r'uv_installer_sha256="[0-9a-f]{64}"', script)
    assert "sha256sum" in script and "shasum -a 256" in script
    assert 'sh "$installer"' in script
    assert "curl -LsSf" in script
    assert "| \\\n    env UV_INSTALL_DIR" not in script
