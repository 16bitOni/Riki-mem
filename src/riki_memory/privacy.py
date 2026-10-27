"""Scrub obvious secrets from an exchange before it is stored or sent to a model.

The extractor is told never to save secrets, but the turn log keeps the raw text and the
extraction call sends it to the model provider, so secrets are removed before either.
Conservative on purpose: only values that are plainly secrets (a password or PIN
stated as such, OTP/CVV codes, API keys, card numbers that pass the Luhn check,
Aadhaar-format numbers). Everything else is left exactly as written.
"""

import re

REDACTED = "[redacted]"

# "the password is Tiger@4521", "PIN: 4821", "otp 552901". Not "pin code" (an Indian postal code).
_LABELLED = re.compile(
    r"\b(pass(?:word|code|phrase)?|passwd|pwd|pin(?!\s*-?\s*code)|otp|cvv|cvc|api[\s_-]?key|"
    r"secret(?:\s+key)?|access\s+token|auth\s+token)"
    r"(\s*(?:is|was|=|:|-)\s*|\s+)"
    r"(?!is\b|was\b)([^\s,;]+)",
    re.IGNORECASE,
)


def _labelled(match: re.Match) -> str:
    """A value that looks like a secret: has a digit or symbol, or follows ':' / '='.
    shortcut: a letters-only password said as "my password is sunshine" is kept, because
    "the password is weak/changed/expired" is far more common; extend if users report it."""
    label, sep, value = match.groups()
    secretish = any(not c.isalpha() for c in value.rstrip(".!?")) or sep.strip() in (":", "=")
    return f"{label}{sep}{REDACTED}" if secretish else match.group(0)


_KEYS = re.compile(
    r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"  # OpenAI/Stripe-style keys
    r"|\bAKIA[0-9A-Z]{16}\b"  # AWS access key id
    r"|\bgh[pousr]_[A-Za-z0-9]{30,}"  # GitHub tokens
    r"|\bxox[abprs]-[A-Za-z0-9-]{10,}"  # Slack tokens
)
_CARD = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
_AADHAAR = re.compile(r"\b\d{4}\s\d{4}\s\d{4}\b")


def _luhn(digits: str) -> bool:
    total, odd = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if odd:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
        odd = not odd
    return total % 10 == 0


def _card(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    return REDACTED if 13 <= len(digits) <= 19 and _luhn(digits) else match.group(0)


def redact_secrets(text: str) -> str:
    if not text:
        return text
    text = _LABELLED.sub(_labelled, text)
    text = _KEYS.sub(REDACTED, text)
    text = _CARD.sub(_card, text)
    return _AADHAAR.sub(REDACTED, text)
