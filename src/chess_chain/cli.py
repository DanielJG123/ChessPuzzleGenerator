"""Command-line interface for Chess Chain."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated, Any

import chess
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from rich.console import Console
from rich.json import JSON
from rich.table import Table

from chess_chain.analysis.cache import DiskKeyValueCache
from chess_chain.analysis.cheap import analyze_cheap_puzzle, stream_cheap_analysis
from chess_chain.analysis.collision import (
    CollisionRequest,
    FinalOccupancyCollisionAnalyzer,
)
from chess_chain.analysis.deep import (
    analyze_deep_puzzle,
    estimate_engine_calls,
    ply_to_deep_record,
)
from chess_chain.analysis.engine import (
    EngineAdapter,
    EngineConfig,
    EngineUnavailableError,
    engine_fingerprint_from_identity,
)
from chess_chain.analysis.forcedness import forcedness_config_from_settings
from chess_chain.analysis.ply import ply_from_record
from chess_chain.analysis.policies import (
    StageEndPolicy,
    evaluate_predicate,
    get_policy,
)
from chess_chain.analysis.records import (
    collision_from_record,
    collision_to_record,
    variation_to_record,
)
from chess_chain.analysis.schema import COLLISION_SCHEMA, DEEP_PLY_SCHEMA, VARIATIONS_SCHEMA
from chess_chain.analysis.summarize import summarize_puzzle
from chess_chain.analysis.variations import (
    TerminalVariationSummary,
    analyze_variations,
    enumerate_accepted_lines,
    estimate_scope_along_reference,
)
from chess_chain.config import Settings, get_settings
from chess_chain.domain.position import PositionFingerprint
from chess_chain.ingest.download import download_file
from chess_chain.ingest.pipeline import ingest_puzzles
from chess_chain.ingest.sample import stream_sample_rows
from chess_chain.reporting.stats import compute_stats
from chess_chain.storage.catalog import activate_dataset, ensure_catalog
from chess_chain.storage.manifest import is_download_complete
from chess_chain.storage.repository import PuzzleRepository

app = typer.Typer(no_args_is_help=True, help="Chess Chain puzzle sequence generator.")
db_app = typer.Typer(no_args_is_help=True, help="Dataset download and sampling commands.")
app.add_typer(db_app, name="db")
analyze_app = typer.Typer(no_args_is_help=True, help="Chess analysis tiers (stage 3).")
app.add_typer(analyze_app, name="analyze")

console = Console()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    # Re-bind handlers on every call so repeated invocations (tests, embedded
    # use) never write to a previously captured/closed stream.
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
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


def _open_analysis_repo(config: Path | None) -> tuple[Settings, PuzzleRepository]:
    settings = _resolve_settings(config)
    processed = settings.data_processed_dir
    assert processed is not None
    con = ensure_catalog(processed)
    return settings, PuzzleRepository(con)


def _fetch_puzzle_row(repo: PuzzleRepository, puzzle_id: str) -> dict[str, Any]:
    row = repo.fetch_puzzle(puzzle_id)
    if row is None:
        raise typer.BadParameter(f"puzzle {puzzle_id!r} not found in the active dataset")
    return row


@analyze_app.command("cheap")
def analyze_cheap_cmd(
    where: Annotated[
        str,
        typer.Option("--where", help="SQL WHERE clause over the puzzles view."),
    ] = "",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output Parquet path."),
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
    """Run pure python-chess structural analysis over the selected puzzles."""
    _configure_logging(verbose)
    settings, repo = _open_analysis_repo(config)
    try:
        count = repo.count_where(where)
        console.print(
            f"[bold]Estimated scope:[/bold] {count} puzzles (cheap tier; no engine required)"
        )
        destination = (output or settings.analysis_dir / "cheap" / "plies.parquet").resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        plies = stream_cheap_analysis(
            rows=repo.stream_analysis_rows(where=where),
            output_path=destination,
        )
        console.print(f"[green]Wrote[/green] {plies} ply records to {destination}")
    finally:
        repo.close()


@analyze_app.command("deep")
def analyze_deep_cmd(
    puzzle_id: Annotated[str, typer.Option("--puzzle-id", help="Puzzle id.")],
    engine_path: Annotated[
        str | None,
        typer.Option("--engine-path", help="Override the configured engine path."),
    ] = None,
    multipv: Annotated[
        int | None,
        typer.Option("--multipv", min=1, help="Override MultiPV."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output Parquet path."),
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
    """Run MultiPV engine analysis for one puzzle (cached by FEN + engine config)."""
    _configure_logging(verbose)
    settings, repo = _open_analysis_repo(config)
    engine: EngineAdapter | None = None
    try:
        row = _fetch_puzzle_row(repo, puzzle_id)
        start_fen = str(row["start_fen"])
        solution_uci = tuple(str(move) for move in row["solution_uci"])
        engine_config = EngineConfig(
            path=engine_path if engine_path is not None else settings.engine_path,
            threads=settings.engine_threads,
            hash_mb=settings.engine_hash_mb,
            movetime_ms=settings.engine_movetime_ms,
            depth=settings.engine_depth,
            nodes=settings.engine_nodes,
            multipv=multipv or settings.engine_multipv,
        )
        engine = EngineAdapter(engine_config)
        name, version = engine.engine_identity()
        engine_fingerprint = engine_fingerprint_from_identity(
            budget=engine_config.budget_key(), name=name, version=version
        )
        cache = DiskKeyValueCache(settings.analysis_dir / "engine_cache")
        total, cached = estimate_engine_calls(
            start_fen=start_fen,
            solution_uci=solution_uci,
            cache=cache,
            engine_fingerprint=engine_fingerprint,
        )
        console.print(
            f"[bold]Engine:[/bold] {name} {version} | budget={engine_config.budget_key()}"
        )
        console.print(
            f"[bold]Estimated engine calls:[/bold] {total} positions ({cached} already cached)"
        )
        forcedness = forcedness_config_from_settings(settings)
        plies, meta = analyze_deep_puzzle(
            puzzle_id=puzzle_id,
            start_fen=start_fen,
            solution_uci=solution_uci,
            solver_colour=str(row["solver_colour"]),
            engine=engine,
            cache=cache,
            forcedness=forcedness,
            engine_fingerprint=engine_fingerprint,
        )
        destination = (output or settings.analysis_dir / "deep" / f"{puzzle_id}.parquet").resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        records = [
            ply_to_deep_record(
                puzzle_id=puzzle_id,
                source=str(row["source"]),
                transform_version=int(row["transform_version"]),
                ply=ply,
                engine_meta=meta,
            )
            for ply in plies
        ]
        with pq.ParquetWriter(destination, DEEP_PLY_SCHEMA, compression="zstd") as writer:
            writer.write_table(pa.Table.from_pylist(records, schema=DEEP_PLY_SCHEMA))
        console.print(f"[green]Wrote[/green] {len(records)} deep ply records to {destination}")
        for ply in plies:
            forcedness_value = ply.forcedness.value if ply.forcedness is not None else "?"
            console.print(
                f"  ply {ply.ply_index}: {ply.uci} forcedness={forcedness_value} "
                f"legal={ply.legal_move_count} plausible={ply.plausible_count}"
            )
    except EngineUnavailableError as exc:
        console.print(f"[red]Engine unavailable:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    finally:
        if engine is not None:
            engine.close()
        repo.close()


@analyze_app.command("variations")
def analyze_variations_cmd(
    puzzle_id: Annotated[str, typer.Option("--puzzle-id", help="Puzzle id.")],
    end_policy: Annotated[
        str,
        typer.Option("--end-policy", help="Stage-end policy name."),
    ] = "checkmate",
    acceptance: Annotated[
        str,
        typer.Option("--acceptance", help="intended_only | all_legal | plausible"),
    ] = "intended_only",
    max_plies: Annotated[
        int | None,
        typer.Option("--max-plies", min=1, help="Line length limit."),
    ] = None,
    max_nodes: Annotated[
        int | None,
        typer.Option("--max-nodes", min=1, help="Node budget."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output Parquet path."),
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
    """Enumerate accepted lines and verify the stage-end policy over endpoints."""
    _configure_logging(verbose)
    settings, repo = _open_analysis_repo(config)
    try:
        row = _fetch_puzzle_row(repo, puzzle_id)
        policy = get_policy(end_policy)
        ply_limit = max_plies or settings.variation_max_plies
        node_limit = max_nodes or settings.variation_max_nodes
        start_fen = str(row["start_fen"])
        intended = tuple(str(move) for move in row["solution_uci"])
        board = chess.Board(start_fen)
        estimate = estimate_scope_along_reference(board, intended, ply_limit)
        console.print(
            f"[bold]Estimated scope:[/bold] up to ~{estimate} accepted lines "
            f"(max_plies={ply_limit}, max_nodes={node_limit})"
        )
        summary = analyze_variations(
            puzzle_id=puzzle_id,
            start_fen=start_fen,
            solution_uci=intended,
            end_policy=policy,
            acceptance=acceptance,
            max_plies=ply_limit,
            max_nodes=node_limit,
        )
        destination = (
            output or settings.analysis_dir / "variations" / f"{puzzle_id}_{policy.name}.parquet"
        ).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with pq.ParquetWriter(destination, VARIATIONS_SCHEMA, compression="zstd") as writer:
            writer.write_table(
                pa.Table.from_pylist([variation_to_record(summary)], schema=VARIATIONS_SCHEMA)
            )
        console.print(f"[green]Wrote[/green] variation summary to {destination}")
        console.print(
            f"lines={summary.line_count} "
            f"exact_position_invariant={summary.exact_position_invariant} "
            f"occupancy_invariant={summary.occupancy_invariant} "
            f"endpoints_verified={summary.endpoints_verified} "
            f"{summary.verification_note or ''}"
        )
    finally:
        repo.close()


def _endpoint_satisfies(policy: StageEndPolicy, board: chess.Board, length: int) -> bool:
    if policy.kind == "CHECKMATE":
        return board.is_checkmate()
    if policy.kind == "TRANSITION_READY":
        return True
    if policy.kind == "EXPLICIT_PREDICATE":
        return evaluate_predicate(board, policy.predicate or "")
    if policy.kind == "FIXED_PLIES":
        return length == (policy.fixed_plies or -1)
    return False


def _accepted_lines_for_policy(
    board: chess.Board,
    intended: tuple[str, ...],
    accepted_policy: str,
    end_policy: StageEndPolicy,
    max_plies: int,
    max_nodes: int,
) -> tuple[tuple[str, ...], ...]:
    if accepted_policy == "intended_only":
        return (intended,)
    lines, _ = enumerate_accepted_lines(
        board,
        intended_uci=intended,
        acceptance="all_legal",
        max_plies=max_plies,
        max_nodes=max_nodes,
    )
    accepted: list[tuple[str, ...]] = []
    for line in lines:
        current = board.copy()
        for uci in line:
            current.push(chess.Move.from_uci(uci))
        if _endpoint_satisfies(end_policy, current, len(line)):
            accepted.append(line)
    return tuple(accepted)


@analyze_app.command("final-collision")
def analyze_final_collision_cmd(
    puzzle_id: Annotated[str, typer.Option("--puzzle-id", help="Puzzle id.")],
    scope: Annotated[
        str | None,
        typer.Option("--scope", help="exact_plies | up_to_plies | plausible_horizon"),
    ] = None,
    max_depth: Annotated[
        int | None,
        typer.Option("--max-depth", min=1, help="Line length limit."),
    ] = None,
    max_nodes: Annotated[
        int | None,
        typer.Option("--max-nodes", min=1, help="Node budget."),
    ] = None,
    target_occupancy: Annotated[
        str | None,
        typer.Option("--target-occupancy", help="Hex occupancy mask override."),
    ] = None,
    accepted_policy: Annotated[
        str,
        typer.Option("--accepted-policy", help="intended_only | end_policy_lines"),
    ] = "intended_only",
    end_policy: Annotated[
        str,
        typer.Option("--end-policy", help="Stage-end policy name."),
    ] = "checkmate",
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output Parquet path."),
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
    """Lazily check a final candidate for unintended-occupancy collisions."""
    _configure_logging(verbose)
    settings, repo = _open_analysis_repo(config)
    try:
        row = _fetch_puzzle_row(repo, puzzle_id)
        start_fen = str(row["start_fen"])
        intended = tuple(str(move) for move in row["solution_uci"])
        board = chess.Board(start_fen)
        scope_name = scope or settings.collision_scope
        depth = max_depth or settings.collision_max_depth
        nodes = max_nodes or settings.collision_max_nodes
        target = int(target_occupancy, 16) if target_occupancy else int(row["final_occupancy"])
        policy = get_policy(end_policy)
        accepted_lines = _accepted_lines_for_policy(
            board, intended, accepted_policy, policy, depth, nodes
        )
        console.print(
            f"[bold]Search scope:[/bold] {scope_name}, max_depth={depth}, "
            f"max_nodes={nodes}, target={target:#x}, accepted_lines={len(accepted_lines)}"
        )
        request = CollisionRequest(
            puzzle_id=puzzle_id,
            source_version=int(row["transform_version"]),
            presented_fingerprint=PositionFingerprint.from_board(board).to_key(),
            target_occupancy=target,
            accepted_line_policy=accepted_policy,
            stage_end_policy=policy.key(),
            scope=scope_name,
            max_depth=depth,
            max_nodes=nodes,
            pruning_config="transposition",
            analyser_version=settings.collision_analyser_version,
        )
        cache = DiskKeyValueCache(settings.analysis_dir / "collision_cache")
        if cache.has(request.cache_key(cache)):
            console.print("[yellow]Cache hit[/yellow] for this scope/configuration")
        analyzer = FinalOccupancyCollisionAnalyzer(cache=cache)
        result = analyzer.analyze(
            request=request,
            board=board,
            accepted_lines=accepted_lines,
            move_filter=None,
        )
        if result.cached:
            console.print("[yellow]Result served from cache[/yellow]")
        console.print(
            f"status={result.status.value} nodes={result.nodes_searched} "
            f"verified_acceptance_occupancy={result.verified_acceptance_occupancy}"
        )
        if result.witness_line:
            console.print(
                f"witness: {' '.join(result.witness_line)} (depth {result.witness_depth})"
            )
        if result.note:
            console.print(f"note: {result.note}")
        destination = (
            output or settings.analysis_dir / "collisions" / f"{puzzle_id}.parquet"
        ).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        record = collision_to_record(result, cache_key=request.cache_key(cache))
        with pq.ParquetWriter(destination, COLLISION_SCHEMA, compression="zstd") as writer:
            writer.write_table(pa.Table.from_pylist([record], schema=COLLISION_SCHEMA))
        console.print(f"[green]Persisted[/green] to {destination}")
    finally:
        repo.close()


@analyze_app.command("summarize")
def analyze_summarize_cmd(
    puzzle_id: Annotated[str, typer.Option("--puzzle-id", help="Puzzle id.")],
    include_collision: Annotated[
        bool,
        typer.Option("--include-collision", help="Include final-collision status."),
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
    """Assemble the product-suitability summary for one puzzle."""
    _configure_logging(verbose)
    settings, repo = _open_analysis_repo(config)
    try:
        row = _fetch_puzzle_row(repo, puzzle_id)
        start_fen = str(row["start_fen"])
        intended = tuple(str(move) for move in row["solution_uci"])
        solver_colour = str(row["solver_colour"])
        cheap = analyze_cheap_puzzle(
            puzzle_id=puzzle_id,
            start_fen=start_fen,
            solution_uci=intended,
            solver_colour=solver_colour,
        )
        deep = None
        deep_path = settings.analysis_dir / "deep" / f"{puzzle_id}.parquet"
        if deep_path.is_file():
            deep = [ply_from_record(record) for record in pq.read_table(deep_path).to_pylist()]
        variation = None
        variation_paths = sorted(
            (settings.analysis_dir / "variations").glob(f"{puzzle_id}_*.parquet")
        )
        if variation_paths:
            variation = TerminalVariationSummary.from_dict(
                pq.read_table(variation_paths[0]).to_pylist()[0]
            )
        collision = None
        if include_collision:
            collision_path = settings.analysis_dir / "collisions" / f"{puzzle_id}.parquet"
            if collision_path.is_file():
                collision = collision_from_record(pq.read_table(collision_path).to_pylist()[0])
        summary = summarize_puzzle(
            puzzle_id=puzzle_id,
            outcome=str(row["outcome"]),
            cheap_plies=cheap,
            deep_plies=deep,
            variation=variation,
            collision=collision,
        )
        console.print(JSON(json.dumps(summary, sort_keys=True, indent=2)))
    finally:
        repo.close()


def main() -> None:
    app()


if __name__ == "__main__":
    main()
