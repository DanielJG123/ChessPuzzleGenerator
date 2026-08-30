# Chess Chain

Generator for **physical sequences of linked chess puzzles**: printable puzzle chains where each solved endpoint plus a visible transition instruction leads to the next start position.

## Product constraints

1. The player sees the first chess position and every written transition instruction from the start.
2. There are no concealed codes, digital reveals, or staged instructions.
3. A solved endpoint `F_i` plus a visible transition `T_i` produces the next start `S_(i+1)`.
4. The player moves both colours.
5. The mechanical checker recognises occupied squares only.
6. The final puzzle ends in checkmate.
7. The project must eventually generate many unique sequences at target difficulty levels.

## Quick start

Requires Python 3.12+.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

Run checks:

```bash
ruff check src tests
ruff format --check src tests
mypy src
pytest
```

Download the official Lichess puzzle export (large; ~290 MB compressed):

```bash
chess-chain db download
```

Create a small development sample (after download, or from any complete `.csv.zst` source):

```bash
chess-chain db sample --rows 10000
```

Ingest the export into processed Parquet + DuckDB catalog (idempotent, resumable):

```bash
chess-chain db ingest --input data/raw/lichess_db_puzzle.csv.zst --output data/processed
```

Report data-quality statistics over the processed dataset:

```bash
chess-chain db stats
```

Stage 3 chess analysis tiers (cheap needs no engine; deep needs a configured
Stockfish-compatible UCI executable, see `config/default.yaml`):

```bash
chess-chain analyze cheap --where "rating >= 1800"
chess-chain analyze deep --puzzle-id 00sHx
chess-chain analyze variations --puzzle-id 00sHx --end-policy checkmate
chess-chain analyze final-collision --puzzle-id 00sHx --scope up_to_plies
chess-chain analyze summarize --puzzle-id 00sHx --include-collision
```

Stage 4 puzzle similarity organiser (ranked candidate pairs):

```bash
chess-chain similarity build-index
chess-chain similarity query --puzzle-id 00sHx --top 50 --end-policy transition_ready
chess-chain similarity explain --from 00sHx --to 00sJ9
```

Stage 5 deterministic transition language and synthesiser (typed AST, executed
and verified, then rendered from templates; no LLM writes instructions):

```bash
chess-chain transition synthesize --from 00sHx --to 00sJ9 --end-policy transition_ready --output data/processed/analysis/transition.json
chess-chain transition verify data/processed/analysis/transition.json
chess-chain transition explain-leakage data/processed/analysis/transition.json
```

Stage 6 complete unique puzzle sequences (versioned `SequenceRequest` YAML,
on-demand neighbour retrieval, beam search with backtracking, lazy
final-collision validation, registry):

```bash
chess-chain sequence generate --config request.yaml
chess-chain sequence validate data/processed/analysis/sequence/SEQ-1.json
chess-chain sequence explain  data/processed/analysis/sequence/SEQ-1.json
chess-chain sequence reproduce --product-id SEQ-1
```

Stage 7 final integration: provisional difficulty estimate, product rendering,
independent validation and the private visual review report:

```bash
chess-chain rating estimate-simple sequence.json
chess-chain product render sequence.json
chess-chain product validate sequence.json --explain-failure
chess-chain product review sequence.json
```

### Shortest successful workflow (all offline, fixture-sized)

```bash
# 1. A tiny CSV.ZST dataset (see tests/test_sequence_generation.py for the format)
chess-chain db ingest --input data/raw/tiny.csv.zst --output data/processed

# 2. Generate one product
cat > request.yaml <<'EOF'
product_id: SEQ-1
num_stages: 3
rating_min: 1400
rating_max: 1600
acceptance: intended_only
max_plies: 4
collision_scope: up_to_plies
collision_max_depth: 1
random_seed: 7
EOF
chess-chain sequence generate --config request.yaml

# 3. Render the product (player instructions, spec, QA, private review HTML)
chess-chain product render data/processed/analysis/sequence/SEQ-1.json

# 4. Independently re-verify the product from its canonical specification
chess-chain product validate data/processed/analysis/sequence/SEQ-1.json
```

The shortest fully scripted workflow from an empty dataset to a rendered,
independently validated product is exercised by
`tests/test_product_stage7.py::test_cli_rating_and_product_workflow`.

A minimal `request.yaml`:

```yaml
product_id: SEQ-1
num_stages: 3
rating_min: 1500
rating_max: 1700
target_rating: 1600
min_popularity: 10
allowed_intermediate_policies: [transition_ready]
collision_scope: up_to_plies
collision_max_depth: 12
max_transition_actions: 4
random_seed: 7
```

Override paths via `config/default.yaml` or environment variables prefixed with `CHESS_CHAIN_`.

## Implemented stages

- **Stage 1:** project structure, configuration, and an idempotent Lichess
  puzzle downloader with streamed sampling.
- **Stage 2:** typed domain structures, python-chess replay of the source
  convention (`presented_start = Board(FEN) after Moves[0]`,
  `solution = Moves[1:]`), a streaming Parquet ingestion pipeline with
  quarantine, stage-local piece identities, canonical fingerprints, DuckDB
  catalog/repository, and a stats command.
- **Stage 3:** chess analysis tiers (cheap / deep / variation / final-collision),
  per-ply forcedness and ambiguity records, versioned stage-end policies,
  and a reusable final-only occupancy-collision analyser with a persistent
  cache. The mechanical validator is only used once, on the final puzzle.
- **Stage 4:** a modular puzzle similarity organiser that retrieves ranked
  candidate pairs `(i, j)` where the accepted endpoints of `i` can become the
  presented start of `j` with short, low-leakage physical instructions: layered
  distances (occupancy Hamming, 12-bitboard placement, material/inventory),
  minimum-cost physical piece assignment (SciPy Hungarian), conservative
  board symmetries reconstructed and validated with python-chess, castling /
  en-passant rights carried through comparisons, bounded grouped retrieval with
  an optional accelerated backend, and version-fingerprinted index persistence.
- **Stage 5:** a deterministic transition language and synthesiser. A typed,
  JSON-serialisable AST (selectors, destinations, conditions, actions) is
  executed and verified by the executor - the only semantic authority - which
  detects ambiguity, missing selectors, off-board destinations, collisions,
  duplicate piece use and inventory violations. Typed `StageBoundaryGuard`s
  (checkmate / transition-ready / explicit predicate / fixed plies) are
  evaluated deterministically with per-policy clue disclosures. Clue and
  leakage accounting covers no-wrap relative-vector geometry
  (`(8-|dx|)*(8-|dy|)` origins, `log2(64/N)` bits), conditional
  entropy `H(before)-H(after)` and the epistemic-anchor rule (uncertainty
  before solving, never "did the piece move?"). Synthesis searches bounded
  action sequences (transforms first, then similarity-assignment-derived
  relative operations) and ranks by correctness, invalidity on
  initial/prefix/wrong states, low reconstructability, low leakage and few
  actions. Persisted `transition.json` documents store the AST plus
  machine-verifiable before/after FENs; `transition verify` re-executes the
  AST so a renderer change can never alter semantics.

- **Stage 6:** complete unique puzzle sequences. A versioned `SequenceRequest`
  (stage count, rating band/spread, popularity/play gates, themes, solver
  uniqueness and opponent forcedness, intermediate `StageEndPolicy` values,
  final-only collision status/scope, transition budget, inventory profile,
  exact-journey preference, checkmate finale, seed, exclusions, search knobs)
  drives a deterministic generator: puzzles are nodes and synthesised
  transitions are directed edges retrieved on demand (no global graph);
  `NeighbourGraph` reuses the similarity index + transition synthesis and caches
  edge attempts by config fingerprint; `BeamSearchSearcher` (behind a
  `SequenceSearcher` protocol) is a bounded, reproducible beam search with
  backtracking; finalists are ranked by cached features and collision-checked
  lazily in rank order (never for intermediate nodes) against the persistent
  final-collision cache. Outputs: canonical `sequence@1` JSON (request,
  fingerprint, stages with accepted branches/endpoint FENs/transition ASTs,
  machine validation report, search report, player-facing representation,
  private manufacturing/QA data), a local manufactured-product registry with an
  optional `forbid_puzzle_reuse` policy, and deterministic reproduction from
  the same seed and configuration.
- **Stage 7:** final integration. A replaceable `RatingEstimator` boundary with
  a `SimpleAverageRatingEstimator` placeholder (arithmetic mean of the stage
  source ratings; strict missing-rating failure; labelled **"Provisional
  source-rating average"**; no fabricated uncertainty interval) stored under a
  forward-compatible `method@estimator_version` key. `product render` emits
  five strictly-separated artifacts: `player_instructions.md` (player-safe
  only), `manufacturing_spec.json` (canonical product specification with the
  final mechanical occupancy target and bounded collision report),
  `qa_report.md`, a self-contained private `sequence_review.html` (python-chess
  SVG + Jinja2; intended replay, accepted branches, transition before/after,
  guard/leakage/forcedness, final-only collision info) and a `review_notes.yaml`
  template. `product validate` reconstructs and independently re-verifies every
  hard constraint from the specification with `--explain-failure`. Config
  precedence is explicit and tested: CLI > env > user YAML > defaults. CI runs
  the offline suite (unit, mypy, ruff, fast fixtures) without heavy downloads
  or Stockfish.

See `docs/architecture.md` for module boundaries, the data schema, the
derived-field inventory, stage-end policies and the bounded meaning of
`collision_free_within_scope`. Per-stage references: `docs/transitions.md`,
`docs/configuration.md`, `docs/contributing.md`.
