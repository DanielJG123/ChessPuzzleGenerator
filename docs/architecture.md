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
| `src/chess_chain/ingest/` | Downloader, streamed sampling, Parquet pipeline |
| `src/chess_chain/storage/` | Manifests, schemas, DuckDB catalog and repository |
| `src/chess_chain/reporting/` | Data-quality statistics |
| `config/` | Default YAML settings |
| `data/raw/` | Immutable downloaded sources |
| `data/interim/` | Intermediate processing artefacts |
| `data/processed/` | Parquet datasets, DuckDB catalog |
| `data/samples/` | Small streamed development subsets |
| `tests/` | Offline unit, integration and property tests |
| `docs/` | Project documentation |

## Module boundaries (implemented and planned)

| Module | Stage 1 | Stage 2 | Later stages |
|--------|---------|---------|--------------|
| `cli` | `db download` / `db sample` | `db ingest` / `db stats` | Additional commands |
| `config` | YAML + env settings | Ingest settings | Analysis parameters |
| `domain` | Constraint documentation | Core types (stage 2) | Extended types |
| `ingest` | Download and sampling | Replay + Parquet pipeline | Puzzle normalisation |
| `chess` | — | Replay, fingerprints, material | Board helpers |
| `storage` | Download manifests | Schemas, catalog, repository | More tables |
| `similarity` | — | — | Puzzle similarity |
| `transitions` | — | — | Transition generation |
| `sequence` | — | — | Sequence search |
| `rating` | — | — | Difficulty estimation |
| `reporting` | — | Stats | Export and reports |

## Configuration

`config/default.yaml` defines data paths, download/ingest settings (batch size,
row-group size, rating bucket width, transformation version) and reserved
analysis defaults. Paths are resolved relative to the project root unless
absolute.

