"""Entry point: ``python -m shift_aso.mearec.generation``."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
