# Architecture

## Purpose

Chess Chain builds printable chains of linked chess puzzles for physical play.

## Domain invariant

For puzzle `i` with solved endpoint `F_i` and visible transition instruction `T_i`:

```text
T_i(F_i) = S_(i+1)
```

The player always sees `S_1` and every `T_i` from the outset. Transitions are
written instructions applied after solving; there are no hidden codes or
staged reveals.

## Product constraints

| # | Constraint |
|---|------------|
| 1 | First position and all transitions visible from the start |
| 2 | No concealed codes, digital reveals, or staged instructions |
| 3 | `T_i(F_i) = S_(i+1)` |
| 4 | Player moves both colours |
| 5 | Mechanical checker: occupied squares only |
| 6 | Final puzzle ends in checkmate |
| 7 | Eventually generate many unique sequences at target difficulty |

## Stage 1 (download and sampling)

- `chess-chain db download` streams the official Lichess puzzle export to a
  `.part` file, verifies size when known, records a SHA-256 sidecar manifest
  and atomically renames. Raw files become read-only after success.
- `chess-chain db sample --rows N` writes a streamed development sample from a
  completed raw `.csv.zst` file without decompressing the full dataset in
  memory.

## Stage 2 (ingestion and derived core fields)

### Replay convention

Using python-chess for all FEN parsing and move legality:

```text
raw_board       = Board(FEN)
preceding_move  = Moves[0]
presented_start = raw_board after preceding_move   # the position the player sees
solution_moves  = Moves[1:]
final_board     = presented_start after all solution_moves
```

### Domain model (`src/chess_chain/domain/`)

- `SourcePuzzle` — immutable raw source fields.
- `PositionFingerprint` — 12 colour/piece bitboards, 64-bit occupancy mask,
  side to move, castling rights and en-passant square. Move counters are
  ignored, as in standard chess position identity; any further relaxation is an
  explicitly named comparison fingerprint.
- `CanonicalPosition` — a board snapshot with a counter-insensitive
  fingerprint used for equality and replay.
- `StagePieceId` — stage-local physical identity `(colour, piece_type,
  start_square)` assigned from the presented start and tracked through the
  solution (e.g. "the piece that began this puzzle on a8"). No historical game
  identity is assumed.
- `MoveEvent` — one solution ply: side, UCI/SAN, moving piece identity, origin,
  destination, capture info, promotion, check and mate flags.
- `PuzzleCore` — presented start, solution UCI/SAN lists, final position and
  cheap derived metadata.
- `MaterialSignature` — piece counts per colour/type with helpers for
  duplicate-queen and beyond-standard-set detection.

Positions needing physical inventory beyond a normal set (promotions,
duplicate queens, unusual material) are recorded explicitly via the
`has_promotion`, `duplicate_queens` and `unusual_material` flags rather than
silently accepted.

### Ingestion pipeline (`chess-chain db ingest --input X --output data/processed`)

- Reads `.csv`, `.csv.zst` or `.zst` as a stream (zstandard); no full
  decompression, no full-dataset DataFrame in memory.
- Replays every row with python-chess. Malformed FENs, illegal preceding
  moves, illegal solution moves and impossible lines are quarantined to
  `quarantine/part-*.parquet` with a reason; the run never aborts.
- Valid rows are written to hive-partitioned Parquet
  (`puzzles/rating_bucket=NNNN/part-*.parquet`, zstd compression). Rating
  bucket (default width 100) is the physical partition; presented piece count
  is stored as a column, avoiding thousands of tiny files. 64-bit bitboards
  are stored as unsigned integers.
- UCI is the canonical stored move form; SAN is derived for reports.
- Idempotent: a complete dataset with matching transformation version and
  source fingerprint is skipped unless `--force`. Partial outputs live in
  `.staging-*` directories and are never mistaken for final outputs; `--resume`
  continues from the last checkpoint at a safe batch boundary (checkpoints are
  written only after records are flushed to disk).
- A `manifest.json` records the source path, source fingerprint, transformation
  version, row counts and code version so derived data can be invalidated when
  the code or source changes.

### Storage and queries

- `catalog.duckdb` exposes `puzzles` and `quarantined` views over the active
  Parquet dataset plus a `datasets` metadata table.
- `PuzzleRepository` (`src/chess_chain/storage/repository.py`) provides typed
  filters (rating range, rating deviation, popularity, play count, theme, line
  length, checkmate status, piece count, material compatibility) so later
  modules do not spread DuckDB SQL.
- `chess-chain db stats` reports counts, quarantine reasons and distributions
  using SQL aggregation only.

## Stage 3 (chess analysis, endpoint policies, final-only collision analysis)

### Hardware model

The mechanical occupied-square validator is used exactly once, on the final
puzzle of a complete sequence, to open the box. Intermediate puzzles are not
mechanically validated, so expensive unintended-occupancy collision analysis is
only required for puzzles considered as the final stage. Intermediate stages
instead need a reliable visible stopping rule (a stage-end policy). Because the
physical player moves both colours, the stored Lichess principal line is not
sufficient: every ply records forcedness and ambiguity information.

### Special moves

Castling, en passant and promotion are ordinary legal moves through
python-chess (no duplicated rules, no blanket rejection). Their metadata is
handled explicitly:

- castling rights and the en-passant target can make visually identical piece
  arrangements legally different (fingerprints include both);
- castling moves two physical pieces (a four-square occupancy delta);
- en passant removes a pawn from a square other than the destination;
- promotion changes the required physical piece identity/inventory.

Occupancy deltas are computed exactly per move (ordinary move: `{from}`
removed; capture: only `{from}` removed, the destination stays occupied;
castling: four squares change).

### Analysis tiers

1. `cheap` — pure python-chess structural analysis for all ingested puzzles
   (legal/evasion counts, flags, occupancy deltas, `literal_forced`).
2. `deep` — Stockfish MultiPV analysis for shortlisted puzzles, cached by the
   canonical engine FEN plus a complete engine/configuration fingerprint.
   Every result records the engine name/version and budget. The engine is an
   external configured executable (path/threads/hash/nodes/depth/movetime/MultiPV
   in config); it is never vendored.
3. `variation` — bounded accepted-line enumeration and terminal-variation
   analysis under a stage-end policy.
4. `final_collision` — bounded unintended-occupancy search, invoked lazily only
   for final-stage candidates.

### Per-ply records and forcedness

Each ply records whose move it is (solver/opponent), total legal moves, evasions
when in check, whether the intended move is the only legal move, capture /
recapture / check / forcing-check / castling / en-passant / promotion / mate
flags, the exact occupancy delta, the engine rank and evaluation of the intended
move, the gap to the second-best move, the number of plausible moves within the
configured gap, and whether moves transpose to the same resulting position
(always zero at a single ply in standard chess, recorded explicitly).

Named forcedness categories (thresholds in versioned config):

- `literal_forced` — exactly one legal move;
- `unique_best` — intended move uniquely best by the configured gap;
- `practically_forced` — alternatives exist but are all decisively worse;
- `ambiguous` — multiple comparably strong moves;
- `invalid_source_line` — intended move not acceptable under the reference
  analysis (not scored or decisively worse).

### Stage-end policies

A versioned `StageEndPolicy` with metadata; no policy is chosen globally in
code (a sequence may prefer checkmate endings while retaining the others for
corpus coverage):

- `CHECKMATE` — the stage ends at checkmate (clearest visible ending; knowing
  mate is the objective may reduce difficulty).
- `TRANSITION_READY` — the stage ends when the next visible transition's formal
  precondition is uniquely true/executable; full verification is deferred to
  stage 5.
- `EXPLICIT_PREDICATE` — a printed condition ("when the black queen has been
  captured") stored as a formal predicate plus rendered wording.
- `FIXED_PLIES` — the player is told how many half-moves to play.

Terminal-variation analysis enumerates accepted lines under an acceptance
policy, applies them, and records exact terminal FENs and occupancy masks.
Exact-position invariance and occupancy invariance are descriptive facts:
intermediate candidates are not rejected because accepted endpoints differ in
occupancy (later transition synthesis may normalise them into one next start).

### Final-only occupancy-collision analysis

`FinalOccupancyCollisionAnalyzer` verifies that all accepted final lines reach
the same target occupancy mask, then searches for any legal non-accepted line
that reaches that mask within a declared scope (`exact_plies`, `up_to_plies`,
`plausible_horizon`), including earlier prefixes (the mechanism could open as
soon as the mask appears). It uses transposition caching and safe pruning,
records a witness line on collision, and reports `collision_found`,
`collision_free_within_scope` or `inconclusive`.

A bounded search never claims global impossibility: `collision_free_within_scope`
means "no collision found within the declared depth/node/scope budget", nothing
more. Results are persisted by a cache key that includes the puzzle/source
version, presented-start fingerprint, target occupancy mask, accepted-line and
stage-end policies, search scope, maximum depth/nodes, pruning configuration and
the analyser algorithm version. This lets a later sequence generator rank final
candidates, query the cache, analyse only uncached candidates, and stop when it
has enough valid finalists.

### CLI (`chess-chain analyze ...`)

- `analyze cheap --where ...`
- `analyze deep --puzzle-id ...`
- `analyze variations --puzzle-id ... --end-policy ...`
- `analyze final-collision --puzzle-id ... --scope ...`
- `analyze summarize --puzzle-id ... [--include-collision]`

Every expensive command prints an estimated scope before running and makes
cache hits visible.



## Dataset provenance

- **Source:** [Lichess open database](https://database.lichess.org/#puzzles)
  puzzle export (`lichess_db_puzzle.csv.zst`, CC0). Fields:
  `PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags,DailyDate`.
- **Acquisition:** `chess-chain db download` (stage 1) with SHA-256 sidecar
  manifest; raw files are immutable after success.
- **Derivation:** `chess-chain db ingest` (stage 2) records
  `transform_version` in the dataset manifest; bumping it invalidates old
  outputs.

## Directory roles

| Path | Role |
|------|------|
| `src/chess_chain/domain/` | Typed domain structures |
| `src/chess_chain/chess/` | python-chess based replay/fingerprint logic |
| `src/chess_chain/analysis/` | Stage-3 tiers: engine, forcedness, variations, collision |
| `src/chess_chain/ingest/` | Downloader, streamed sampling, Parquet pipeline |
| `src/chess_chain/storage/` | Manifests, schemas, DuckDB catalog and repository |
| `src/chess_chain/reporting/` | Data-quality statistics |
| `config/` | Default YAML settings |
| `data/raw/` | Immutable downloaded sources |
| `data/interim/` | Intermediate processing artefacts |
| `data/processed/` | Parquet datasets, DuckDB catalog, analysis outputs |
| `data/samples/` | Small streamed development subsets |
| `tests/` | Offline unit, integration and property tests |
| `docs/` | Project documentation |

## Module boundaries (implemented and planned)

| Module | Stage 1 | Stage 2 | Stage 3 | Later stages |
|--------|---------|---------|---------|--------------|
| `cli` | `db download` / `db sample` | `db ingest` / `db stats` | `analyze *` | Additional commands |
| `config` | YAML + env settings | Ingest settings | Engine/forcedness settings | Analysis parameters |
| `domain` | Constraint documentation | Core types | Analysis value types | Extended types |
| `ingest` | Download and sampling | Replay + Parquet pipeline | — | Puzzle normalisation |
| `chess` | — | Replay, fingerprints, material | — | Board helpers |
| `storage` | Download manifests | Schemas, catalog, repository | — | More tables |
| `analysis` | — | — | Engine tiers, policies, collision | Deep analysis reuse |
| `similarity` | — | — | — | Puzzle similarity |
| `transitions` | — | — | — | Transition generation |
| `sequence` | — | — | — | Sequence search |
| `rating` | — | — | — | Difficulty estimation |
| `reporting` | — | Stats | Summaries | Export and reports |

## Configuration

`config/default.yaml` defines data paths, download/ingest settings (batch size,
row-group size, rating bucket width, transformation version) and reserved
analysis defaults. Paths are resolved relative to the project root unless
absolute.

