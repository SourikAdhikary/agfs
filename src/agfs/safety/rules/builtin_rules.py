"""Layer 4: Built-in ePCA Commit-Hook State Invariant Rules."""

from __future__ import annotations

import fnmatch
import re

DEFAULT_SECRET_REGEX = (
    r"(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}|"
    r"AIza[0-9A-Za-z\-_]{35}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)

DEFAULT_TEST_SKIP_REGEX = (
    r"(?:@pytest\.mark\.skip\b|@pytest\.mark\.xfail\b|@unittest\.skip\b|pytest\.skip\()"
)


def match_protected_glob(path: str, globs: list[str]) -> str | None:
    """Return the matched glob pattern if `path` matches any protected pattern, else None."""
    norm = path.lstrip("./")
    for pat in globs:
        clean_pat = pat.lstrip("./")
        if fnmatch.fnmatch(norm, clean_pat):
            return pat
        if clean_pat.endswith("/**") and (
            norm == clean_pat[:-3] or norm.startswith(clean_pat[:-2])
        ):
            return pat
    return None


def scan_forbidden_regex(text: str, regex_pattern: str) -> str | None:
    """Return the first matched forbidden substring if `regex_pattern` hits in `text`."""
    rx = re.compile(regex_pattern)
    m = rx.search(text)
    if m:
        return m.group(0)
    return None
