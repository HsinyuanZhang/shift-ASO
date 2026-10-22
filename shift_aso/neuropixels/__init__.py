"""Frozen no-drift replay and case-bound dataset preparation (lazy API)."""

__all__ = ["run"]


def __getattr__(name):
    if name == "run":
        from .core import run
        globals()[name] = run
        return run
    raise AttributeError(name)
