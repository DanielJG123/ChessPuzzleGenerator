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

Puzzle similarity, transition synthesis, sequence construction and rating
estimation are planned but not implemented yet.

See `docs/architecture.md` for module boundaries, the data schema, the
derived-field inventory, stage-end policies and the bounded meaning of
`collision_free_within_scope`.
