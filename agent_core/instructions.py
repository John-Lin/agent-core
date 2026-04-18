from __future__ import annotations

import os
from pathlib import Path

INSTRUCTIONS_FILE = Path(os.getenv("AGENT_INSTRUCTIONS_PATH", "instructions.md"))


def _load_instructions() -> str:
    """Load agent instructions from the instructions file.

    Path resolution order:
      1. ``AGENT_INSTRUCTIONS_PATH`` environment variable, if set.
      2. ``instructions.md`` in the process working directory.

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
