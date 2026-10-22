"""Public MEArec replay and dataset preparation (lazy legacy API)."""

__all__ = ["evaluate"]


def __getattr__(name):
    if name == "evaluate":
        from .evaluate import evaluate
        globals()[name] = evaluate
        return evaluate
    raise AttributeError(name)
