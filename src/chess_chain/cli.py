"""Command-line interface for Chess Chain."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from chess_chain.config import Settings, get_settings
from chess_chain.ingest.download import download_file
from chess_chain.ingest.pipeline import ingest_puzzles
from chess_chain.ingest.sample import stream_sample_rows
from chess_chain.reporting.stats import compute_stats
from chess_chain.storage.catalog import activate_dataset, ensure_catalog
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


@db_app.command("ingest")
def db_ingest(
    input_path: Annotated[
        Path | None,
        typer.Option("--input", "-i", help="Source .csv, .zst or .csv.zst puzzle export."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Processed data directory."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Rebuild the dataset even when up to date."),
    ] = False,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Continue from the last staging checkpoint."),
    ] = False,
    batch_size: Annotated[
        int | None,
        typer.Option("--batch-size", min=1, help="Rows between checkpoints/log lines."),
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
    """Stream a Lichess puzzle export into processed Parquet + DuckDB catalog."""
    _configure_logging(verbose)
    settings = _resolve_settings(config)
    source = (input_path or settings.default_puzzle_path).resolve()
    if not source.is_file():
        raise typer.BadParameter(f"Input file does not exist: {source}")
    processed = settings.data_processed_dir
    assert processed is not None
    destination = (output or processed).resolve()

    if batch_size is not None:
        settings = settings.model_copy(
            update={
                "ingest_batch_size": batch_size,
                "parquet_row_group_rows": min(settings.parquet_row_group_rows, batch_size),
            }
        )

    summary = ingest_puzzles(
        input_path=source,
        output_dir=destination,
        settings=settings,
        force=force,
        resume=resume,
    )
    if summary.skipped:
        console.print(f"[yellow]Dataset already up to date:[/yellow] {summary.dataset_dir}")
    else:
        console.print(
            f"[green]Ingested:[/green] {summary.puzzle_rows} puzzles, "
            f"{summary.quarantine_rows} quarantined in {summary.elapsed_seconds:.1f}s"
        )
        console.print(f"Dataset: {summary.dataset_dir}")

    con = ensure_catalog(destination)
    activate_dataset(con, summary)
    console.print(f"Catalog: {destination / 'catalog.duckdb'}")


@db_app.command("stats")
def db_stats(
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Processed data directory."),
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
    """Report data-quality statistics over the processed dataset."""
    _configure_logging(verbose)
    settings = _resolve_settings(config)
    processed = settings.data_processed_dir
    assert processed is not None
    destination = (output or processed).resolve()

    con = ensure_catalog(destination)
    stats = compute_stats(con)
    con.close()

    for name, frame in stats.items():
        if name == "counts":
            console.print(f"Puzzles: {frame['puzzle_rows'][0]}")
        elif name == "quarantined":
            console.print(f"Quarantined rows: {frame['quarantine_rows'][0]}")
        elif name == "flags":
            console.print(
                f"Flags: promotions={frame['has_promotion'][0]} "
                f"duplicate-queens={frame['duplicate_queens'][0]} "
                f"unusual-material={frame['unusual_material'][0]} "
                f"checkmates={frame['checkmates'][0]}"
            )
        else:
            console.print(f"\n[bold]{name}[/bold]")
            _print_frame(frame, max_rows=20)


def _print_frame(frame: Any, max_rows: int) -> None:
    table = Table(show_header=True, header_style="bold cyan")
    for column in frame.columns:
        table.add_column(str(column))
    for row in frame.head(max_rows).iter_rows():
        table.add_row(*[str(value) for value in row])
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
