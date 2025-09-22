"""Module entrypoint for `python -m orchestrator.cli`"""

from .main import main

if __name__ == "__main__":
    raise SystemExit(main())

