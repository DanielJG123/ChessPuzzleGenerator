# Architecture (stage 1)

## Purpose

Chess Chain builds printable chains of linked chess puzzles for physical play. Stage 1 provides project scaffolding, configuration, and Lichess puzzle dataset ingestion.

## Domain invariant

For puzzle `i` with solved endpoint `F_i` and visible transition instruction `T_i`:

```text
T_i(F_i) = S_(i+1)
```

The player always sees `S_1` and every `T_i` from the outset. Transitions are written instructions applied after solving; there are no hidden codes or staged reveals.

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

## Dataset provenance

- **Source:** [Lichess open database](https://database.lichess.org/#puzzles) puzzle export (`lichess_db_puzzle.csv.zst`, CC0).
- **Acquisition:** `chess-chain db download` streams to a `.part` file, verifies size when known, records SHA-256 in a sidecar manifest, then atomically renames.
- **Immutability:** Raw files under `data/raw/` are not modified after a successful download; samples are derived copies under `data/samples/`.

## Directory roles

| Path | Role |
|------|------|
| `src/chess_chain/` | Application package |
| `config/` | Default YAML settings |
| `data/raw/` | Immutable downloaded sources |
| `data/interim/` | Intermediate processing artefacts |
| `data/processed/` | Analysis-ready outputs |
| `data/samples/` | Small streamed development subsets |
| `tests/` | Offline unit and CLI tests |
| `docs/` | Project documentation |

## Module boundaries (planned)

| Module | Stage 1 | Later stages |
|--------|---------|--------------|
| `cli` | Typer entrypoint, `db download` / `db sample` | Additional commands |
| `config` | YAML + env settings | Analysis parameters |
| `domain` | Constraint documentation | Core types |
| `ingest` | Lichess download and sampling | Puzzle normalisation |
| `chess` | — | Board helpers via python-chess |
| `storage` | Download manifests | Parquet / DuckDB tables |
| `similarity` | — | Puzzle similarity |
| `transitions` | — | Transition generation |
| `sequence` | — | Sequence search |
| `rating` | — | Difficulty estimation |
| `reporting` | — | Export and reports |

## Configuration

`config/default.yaml` defines data paths, the Lichess puzzle URL, download chunk size, and reserved analysis defaults. Paths are resolved relative to the project root unless absolute.
