"""Stage 7: product rendering and independent validation.

The canonical ``sequence@1`` record is materialised into player-safe
instructions, a manufacturing specification, a QA report, a private visual
review HTML and a review-notes template, and can be independently validated by
reconstructing every check from the specification.
"""

from __future__ import annotations

from chess_chain.product.render import (
    ProductArtifacts,
    render_product,
    resolve_estimate,
    write_sequence_with_estimate,
)
from chess_chain.product.validate import (
    ProductIssue,
    ProductValidationReport,
    validate_product,
)

__all__ = [
    "ProductArtifacts",
    "ProductIssue",
    "ProductValidationReport",
    "render_product",
    "resolve_estimate",
    "validate_product",
    "write_sequence_with_estimate",
]
