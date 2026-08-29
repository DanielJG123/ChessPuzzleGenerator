"""Product constraints and core domain types (stage 1 placeholders)."""

from __future__ import annotations

# Product constraints (see docs/architecture.md):
# 1. The player sees the first position and every written transition from the start.
# 2. No concealed codes, digital reveals, or staged instructions.
# 3. A solved endpoint F_i plus visible transition T_i produces S_(i+1).
# 4. The player moves both colours.
# 5. The mechanical checker recognises occupied squares only.
# 6. The final puzzle ends in checkmate.
# 7. The project must eventually generate many unique sequences at target difficulty.

__all__: list[str] = []
