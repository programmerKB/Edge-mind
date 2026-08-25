"""Compatibility facade for the modular :mod:`forecast` package.

Existing scripts can continue importing ``forecasting``; new code should
import the public API directly from ``forecast``.
"""

from forecast import *  # noqa: F403 - compatibility re-export by design
