import regex as re
from rapidfuzz.distance import Levenshtein

class FormatGuard:
    def __init__(self, allow_chars: str | None, pattern: str):
        self.allow_chars = allow_chars
        self.re = re.compile(pattern)

    def clean_chars(self, s: str) -> str:
        if not self.allow_chars:
            return s
        allow_set = set(self.allow_chars)
        return "".join(ch for ch in s if ch in allow_set)

    def is_valid(self, s: str) -> bool:
        return bool(self.re.match(s))


def cer(a: str, b: str) -> float:
    if not b:
        return 0.0 if not a else 1.0
    return Levenshtein.distance(a, b) / max(1, len(b))
