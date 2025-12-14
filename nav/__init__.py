"""
`nav` package: navigation utilities + runnable scripts for LeKiwi.

This package exposes the library-style entrypoints needed by orchestrators
like `sentry.py`, while still allowing `nav/nav.py` to be executed as a script.
"""

from .nav import CombinedNavigator

__all__ = ["CombinedNavigator"]

