"""Read secrets from .env without adding a dependency.

python-dotenv would do this, but it is one more thing to install on a machine
that has to work in front of an interviewer, and the format we need is one
KEY=value per line.

A real environment variable wins over the file, so CI or a shell export can
override it without editing anything.
"""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def load_env(path=ENV_FILE):
    """Return the parsed .env as a dict. Missing file is not an error."""
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        values[k.strip()] = v.strip().strip('"').strip("'")
    return values


def secret(name, hint=""):
    """Environment first, then .env. Raises with instructions rather than a
    KeyError, because the fix is a file the caller has to create."""
    value = os.environ.get(name) or load_env().get(name)
    if not value:
        raise SystemExit(
            f"{name} is not set.\n"
            f"  Put it in {ENV_FILE} as   {name}=your-key-here\n"
            f"  (that file is gitignored; see .env.example)"
            + (f"\n  {hint}" if hint else "")
        )
    return value


if __name__ == "__main__":
    # Check the key is readable without ever printing it.
    for name in ("ROBOFLOW_API_KEY",):
        try:
            v = secret(name)
            print(f"{name}: set, {len(v)} chars, ends with ...{v[-2:]}")
        except SystemExit as exc:
            print(exc)
