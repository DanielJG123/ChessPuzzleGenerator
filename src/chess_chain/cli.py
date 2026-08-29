"""Command-line interface for Chess Chain."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from chess_chain.config import Settings, get_settings
from chess_chain.ingest.download import download_file
from chess_chain.ingest.sample import stream_sample_rows
from chess_chain.storage.manifest import is_download_complete

app = typer.Typer(no_args_is_help=True, help="Chess Chain puzzle sequence generator.")
db_app = typer.Typer(no_args_is_help=True, help="Dataset download and sampling commands.")
app.add_typer(db_app, name="db")

console = Console()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _resolve_settings(config: Path | None) -> Settings:
    if config is None:
        return get_settings()
    return Settings.from_yaml(config)


@db_app.command("download")
def db_download(
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Destination file path."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Re-download even when a valid manifest exists."),
    ] = False,
    no_resume: Annotated[
        bool,
        typer.Option("--no-resume", help="Do not resume partial .part downloads."),
    ] = False,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Path to YAML configuration file."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Download the official Lichess puzzle export."""
    _configure_logging(verbose)
    settings = _resolve_settings(config)
    destination = (output or settings.default_puzzle_path).resolve()

    result = download_file(
        url=settings.lichess_puzzle_url,
        output_path=destination,
        settings=settings,
        force=force,
        resume=not no_resume,
    )
    if result.skipped:
        console.print(f"[green]Already complete:[/green] {result.target_path}")
    else:
        console.print(f"[green]Downloaded:[/green] {result.target_path}")
    console.print(f"Manifest: {result.manifest_path}")
    console.print(f"SHA-256: {result.manifest.sha256}")


@db_app.command("sample")
def db_sample(
    rows: Annotated[
        int | None,
        typer.Option("--rows", min=1, help="Number of data rows."),
    ] = None,
    source: Annotated[
        Path | None,
        typer.Option("--source", help="Compressed source .csv.zst file."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output CSV sample path."),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Path to YAML configuration file."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Create a small streamed development sample from the puzzle export."""
    _configure_logging(verbose)
    settings = _resolve_settings(config)
    row_count = rows if rows is not None else settings.default_sample_rows
    source_path = (source or settings.default_puzzle_path).resolve()
    if not is_download_complete(source_path):
        raise typer.BadParameter(
            f"Source file is missing or incomplete: {source_path}. "
            "Run `chess-chain db download` first or pass --source."
        )

    sample_name = f"lichess_puzzle_{row_count}_rows.csv"
    samples_dir = settings.data_samples_dir
    assert samples_dir is not None
    destination = (output or samples_dir / sample_name).resolve()

    written = stream_sample_rows(
        source_path,
        destination,
        rows=row_count,
        chunk_size=settings.download_chunk_size,
    )
    console.print(f"[green]Sample written:[/green] {destination} ({written} rows)")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
