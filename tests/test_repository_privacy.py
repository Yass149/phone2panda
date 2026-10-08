from __future__ import annotations

import pytest

from phone2panda.privacy import text_findings_bytes


@pytest.mark.parametrize(
    ("secret", "label"),
    [
        ("sk-" + "proj-" + "A" * 28, "OpenAI API key"),
        ("AIza" + "A" * 35, "Google API key"),
        ("xoxb-" + "1" * 12 + "-" + "A" * 24, "Slack token"),
        ("glpat-" + "A" * 24, "GitLab token"),
    ],
)
def test_high_confidence_secrets_are_detected(secret: str, label: str) -> None:
    assert label in text_findings_bytes(secret.encode())


def test_documented_secret_placeholders_are_allowed() -> None:
    placeholders = b"OPENAI_API_KEY=your-key-here\nGITHUB_TOKEN=<set-in-environment>\n"
    assert text_findings_bytes(placeholders) == []
