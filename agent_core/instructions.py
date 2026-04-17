from __future__ import annotations

from pathlib import Path

INSTRUCTIONS_FILE = Path("instructions.md")


def _load_instructions() -> str:
    """Load agent instructions from ``instructions.md`` in the working directory.

    Fails fast with a clear error if the file is missing, so misconfiguration
    is caught immediately at startup.
    """
    try:
        return INSTRUCTIONS_FILE.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Instructions file not found: {INSTRUCTIONS_FILE.resolve()}. "
            "Create or mount instructions.md with the agent system prompt."
        ) from e
