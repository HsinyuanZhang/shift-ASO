"""Current MEArec software CLI; legacy HDF5 replay is explicitly opt-in."""
import os
import sys


def main():
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMBA_NUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    arguments = sys.argv[1:]
    if arguments[:1] == ["legacy"]:
        from .legacy import main as legacy_main
        sys.argv = [sys.argv[0]] + arguments[1:]
        legacy_main()
    else:
        from .software.cli import main as software_main
        software_main(arguments)


if __name__ == "__main__":
    main()
