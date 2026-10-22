"""MEArec data preparation and software methods; evaluate is the legacy HDF5 API."""

__all__ = ["evaluate"]


def __getattr__(name):
    if name == "evaluate":
        from .evaluate import evaluate
        globals()[name] = evaluate
        return evaluate
    raise AttributeError(name)
