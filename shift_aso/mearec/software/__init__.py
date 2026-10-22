"""Current MEArec plain-NEO/EMA API; no bundled observations or winners.

The root CLI sets thread limits before importing this package. Library users
should set their own numeric thread environment before importing NumPy.
"""
from .features import centered_neo, ema
from .detect import detect
from .api import run, calibrate
from .stream import replay
from .search import search

__all__ = ['centered_neo', 'ema', 'detect', 'run', 'calibrate', 'replay', 'search']
