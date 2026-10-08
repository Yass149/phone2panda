"""High-confidence text checks used by repository publication gates."""

import re

_SECRET_PATTERNS = {
    "absolute macOS home path": re.compile("/" + r"Users/[^/\s]+/"),
    "absolute Linux home path": re.compile("/" + r"home/[^/\s]+/"),
    "absolute Windows home path": re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+\\"),
    "AWS access key": re.compile("AK" + r"IA[0-9A-Z]{16}"),
    "GitHub token": re.compile("gh" + r"[pousr]_[A-Za-z0-9_]{20,}"),
    "GitLab token": re.compile("gl" + r"pat-[A-Za-z0-9_-]{20,}"),
    "Google API key": re.compile("AI" + r"za[0-9A-Za-z_-]{35}"),
    "OpenAI API key": re.compile("sk" + r"-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    "Slack token": re.compile("xo" + r"x[baprs]-[A-Za-z0-9-]{20,}"),
    "private key": re.compile("-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def text_findings_bytes(data: bytes) -> list[str]:
    """Return labels for machine-local paths and high-confidence secrets."""

    if b"\0" in data:
        return []
    text = data.decode("utf-8", errors="replace")
    return [label for label, pattern in _SECRET_PATTERNS.items() if pattern.search(text)]
