import os

_FALSY_ENV_VALUES = frozenset({"", "0", "false", "no", "off"})


def env_flag(name: str) -> bool:
    """Return True if env var ``name`` is set to a truthy value.

    Common falsy spellings (empty, "0", "false", "no", "off") are treated as
    disabled so that ``FOO=0`` behaves as users intuitively expect rather than
    as Python's default "non-empty string is truthy" rule.
    """
    raw = os.getenv(name)
    if raw is None:
        return False
    return raw.strip().lower() not in _FALSY_ENV_VALUES
