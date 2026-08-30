"""Transition configuration (re-exported from ``chess_chain.config``).

``TransitionConfig`` and ``DEFAULT_GLOBAL_RULE`` live in ``config.py`` so the
``Settings`` model can build the dataclass without triggering the heavier
``transitions`` package imports (which would be circular).
"""

from __future__ import annotations

from chess_chain.config import DEFAULT_GLOBAL_RULE, TransitionConfig

__all__ = ["DEFAULT_GLOBAL_RULE", "TransitionConfig"]
