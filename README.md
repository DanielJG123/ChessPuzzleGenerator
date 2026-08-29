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

Download the official Lichess puzzle export (large; ~200+ MB compressed):

```bash
chess-chain db download
```

Create a small development sample (after download, or from any complete `.csv.zst` source):

```bash
chess-chain db sample --rows 10000
```

Override paths via `config/default.yaml` or environment variables prefixed with `CHESS_CHAIN_`.

## Stage 1 scope

This stage sets up project structure, configuration, and an idempotent Lichess puzzle downloader with streamed sampling. Puzzle similarity, transitions, sequence search, and rating estimation are planned but not implemented yet.

See `docs/architecture.md` for module boundaries and data layout.
