# Contributor notes

## Adding a transition selector or action

1. Define the node in `src/chess_chain/transitions/ast.py` (dataclass + JSON
   round-trip + a `kind`/discriminator).
2. Implement resolution in `executor.py`: selectors resolve physical pieces
   (missing/ambiguous are errors); actions mutate the board and the tracked
   identities; collision/duplicate/inventory checks must not be bypassed.
3. Add a renderer branch in `render.py` (board-global wording only - never
   "left"/"up").
4. Extend the clue accounting in `leakage.py` if the node adds absolute
   disclosures or vector leakage.
5. Add unit + property tests mirroring `tests/test_transition_executor.py`.

## Adding a similarity backend

Implement `SimilarityBackend` (see `similarity/backend.py`): the exact backend
is the reference result; accelerated backends must return the same ranked
shortlist. Register the name in `get_backend`. Add a comparison test against
`ExactChunkedBackend` on a small corpus.

## Adding an engine adapter

`analysis/engine.py::EngineAdapter` wraps python-chess UCI. Add a
`FakeEngineAdapter` in `tests/conftest.py` for offline tests. The adapter must
expose `engine_identity()`, `analyze()` (returning `EngineAnalysisResult`) and
`close()`. Keep `analysis.engine.path` unconfigured for CI - engine tiers are
never exercised in ordinary CI.

## Adding a rating model

Implement the `RatingEstimator` protocol in `sequence/rating.py`
(`estimator_version` + `estimate(stage_ratings) -> DifficultyEstimate`) and
store results under your own `method@estimator_version` key in
`sequence.difficulty_estimates`. `product render` and `product validate` pick
up new keys automatically without invalidating old records.

## Testing and CI

- Run `pytest tests -q` (offline; no downloads, no Stockfish).
- Run `ruff check src tests`, `ruff format --check src tests`, `mypy src`.
- CI (`.github/workflows/ci.yml`) runs exactly these, plus the fast
  end-to-end fixture. Heavy downloads and Stockfish analysis must not run in
  ordinary CI.
