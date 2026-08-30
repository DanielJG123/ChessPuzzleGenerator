"""Command-line interface for Chess Chain."""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Annotated, Any, cast

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
from chess_chain.domain.position import CanonicalPosition, PositionFingerprint
from chess_chain.ingest.download import download_file
from chess_chain.ingest.pipeline import ingest_puzzles
from chess_chain.ingest.sample import stream_sample_rows
from chess_chain.reporting.stats import compute_stats
from chess_chain.similarity import (
    get_backend,
    load_or_build_index,
    resolve_accepted_endpoints,
    retrieve_with_index,
)
from chess_chain.similarity.assignment import CostWeights
from chess_chain.similarity.candidate import PairCandidate
from chess_chain.similarity.index import build_index, save_index
from chess_chain.similarity.records import candidate_to_record
from chess_chain.similarity.retrieval import explain_pair
from chess_chain.similarity.schema import CANDIDATE_SCHEMA
from chess_chain.similarity.scoring import SimilarityScoreWeights
from chess_chain.storage.catalog import activate_dataset, ensure_catalog
from chess_chain.storage.manifest import is_download_complete
from chess_chain.storage.repository import PuzzleRepository
from chess_chain.transitions.context import build_context_from_row
from chess_chain.transitions.records import (
    build_transition_document,
    load_transition_document,
    write_transition_document,
)
from chess_chain.transitions.synth import SynthesisCandidate, synthesize
from chess_chain.transitions.verify import VerificationResult

app = typer.Typer(no_args_is_help=True, help="Chess Chain puzzle sequence generator.")
db_app = typer.Typer(no_args_is_help=True, help="Dataset download and sampling commands.")
app.add_typer(db_app, name="db")
analyze_app = typer.Typer(no_args_is_help=True, help="Chess analysis tiers (stage 3).")
app.add_typer(analyze_app, name="analyze")
similarity_app = typer.Typer(
    no_args_is_help=True,
    help="Puzzle similarity organiser (stage 4): ranked candidate pairs.",
)
app.add_typer(similarity_app, name="similarity")
transition_app = typer.Typer(
    no_args_is_help=True,
    help="Deterministic transition language and synthesiser (stage 5).",
)
app.add_typer(transition_app, name="transition")
sequence_app = typer.Typer(
    no_args_is_help=True,
    help="Complete unique puzzle sequences (stage 6): generation, validation.",
)
app.add_typer(sequence_app, name="sequence")
rating_app = typer.Typer(
    no_args_is_help=True,
    help="Difficulty estimation (stage 7, provisional placeholder).",
)
app.add_typer(rating_app, name="rating")
product_app = typer.Typer(
    no_args_is_help=True,
    help="Product render / independent validation / private review (stage 7).",
)
app.add_typer(product_app, name="product")

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


def _engine_move_scores(
    *,
    settings: Settings,
    engine_path: str,
    start_fen: str,
    intended: tuple[str, ...],
) -> dict[str, dict[str, int]]:
    """Per-position engine scores along the reference line for plausible
    acceptance (precedence: CLI ``--engine-path`` > ``CHESS_CHAIN_ENGINE_PATH``
    / user YAML > defaults). Uses the persistent engine cache."""
    from chess_chain.analysis.engine import EngineAdapter, EngineConfig

    board = chess.Board(start_fen)
    positions = [board.copy()]
    for uci in intended:
        move = chess.Move.from_uci(uci)
        if move in board.legal_moves:
            board.push(move)
            positions.append(board.copy())
    engine_config = EngineConfig(
        path=engine_path,
        threads=settings.engine_threads,
        hash_mb=settings.engine_hash_mb,
        movetime_ms=settings.engine_movetime_ms,
        depth=settings.engine_depth,
        nodes=settings.engine_nodes,
        multipv=settings.engine_multipv,
    )
    engine = EngineAdapter(engine_config)
    move_scores: dict[str, dict[str, int]] = {}
    for position in positions:
        engine_fen = " ".join(position.fen().split()[:4])
        if engine_fen in move_scores:
            continue
        analyses = engine.analyze(position, multipv=engine_config.multipv)
        scores: dict[str, int] = {}
        for move_info in analyses.moves:
            score = move_info.score_cp
            if score is not None:
                scores[str(move_info.uci)] = int(score)
        move_scores[engine_fen] = scores
    return move_scores


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
    engine_path: Annotated[
        str | None,
        typer.Option(
            "--engine-path",
            help="Override the configured engine path (used for acceptance='plausible').",
        ),
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
        move_scores = None
        if acceptance == "plausible":
            resolved_engine = engine_path if engine_path is not None else settings.engine_path
            if not resolved_engine:
                raise typer.BadParameter(
                    "acceptance='plausible' needs an engine: set analysis.engine.path, "
                    "CHESS_CHAIN_ENGINE_PATH, or pass --engine-path"
                )
            move_scores = _engine_move_scores(
                settings=settings,
                engine_path=resolved_engine,
                start_fen=start_fen,
                intended=intended,
            )
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
            move_scores=move_scores,
            plausible_gap_cp=settings.plausible_gap_cp,
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


def _similarity_weights(settings: Settings) -> tuple[SimilarityScoreWeights, CostWeights]:
    score = SimilarityScoreWeights(**settings.similarity_score_weights)
    cost = CostWeights(**settings.similarity_cost_weights)
    return score, cost


def _absolute_disclosures(candidate: SynthesisCandidate) -> int:
    clue = candidate.verification.clue
    return clue.absolute_source_squares + clue.absolute_destination_squares


def _report_absolute(verification: VerificationResult) -> int:
    clue = verification.clue
    return clue.absolute_source_squares + clue.absolute_destination_squares


def _start_counts(row: dict[str, Any]) -> tuple[int, ...]:
    from chess_chain.domain.pieces import COUNT_NAMES

    return tuple(int(row[f"start_{name}"]) for name in COUNT_NAMES)

    from chess_chain.domain.pieces import COUNT_NAMES

    return tuple(int(row[f"start_{name}"]) for name in COUNT_NAMES)


@similarity_app.command("build-index")
def similarity_build_index_cmd(
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Directory for the persisted index."),
    ] = None,
    where: Annotated[
        str,
        typer.Option("--where", help="SQL WHERE clause over the puzzles view."),
    ] = "",
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Path to YAML configuration file."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Build (and persist) the similarity candidate index over the active dataset."""
    _configure_logging(verbose)
    settings, repo = _open_analysis_repo(config)
    try:
        index = build_index(repo, version=settings.similarity_version, where=where)
        destination = (output or settings.similarity_dir).resolve()
        save_index(destination, index)
        console.print(f"[green]Indexed[/green] {len(index.rows)} puzzles -> {destination}")
        console.print(f"Fingerprint: {index.fingerprint()[:16]}...")
    finally:
        repo.close()


def _print_candidate_table(candidates: list[PairCandidate]) -> None:
    if not candidates:
        console.print("[yellow]No candidates found for the given filters.[/yellow]")
        return
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("rank")
    table.add_column("next")
    table.add_column("symmetry")
    table.add_column("d_occ b/m/w")
    table.add_column("cost b/m/w")
    table.add_column("moves")
    table.add_column("add/rem")
    table.add_column("score")
    table.add_column("reasons")
    for rank, candidate in enumerate(candidates, start=1):
        reasons = " | ".join(candidate.reasons_for_rejection) or "-"
        table.add_row(
            str(rank),
            candidate.next_puzzle_id,
            candidate.applied_symmetry,
            (f"{candidate.d_occ_best:.0f}/{candidate.d_occ_mean:.1f}/{candidate.d_occ_worst:.0f}"),
            (f"{candidate.cost_best:.1f}/{candidate.cost_mean:.1f}/{candidate.cost_worst:.1f}"),
            str(candidate.total_moves),
            f"{candidate.total_additions}/{candidate.total_removals}",
            f"{candidate.transition_complexity_score:.2f}",
            reasons,
        )
    console.print(table)


@similarity_app.command("query")
def similarity_query_cmd(
    puzzle_id: Annotated[str, typer.Option("--puzzle-id", help="Source puzzle id.")],
    rating_min: Annotated[
        int | None,
        typer.Option("--rating-min", help="Minimum target rating band."),
    ] = None,
    rating_max: Annotated[
        int | None,
        typer.Option("--rating-max", help="Maximum target rating band."),
    ] = None,
    top: Annotated[
        int | None,
        typer.Option("--top", min=1, help="Number of ranked candidates to keep."),
    ] = None,
    end_policy: Annotated[
        str,
        typer.Option("--end-policy", help="Stage-end policy of the source puzzle."),
    ] = "transition_ready",
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="exact | accelerated (override config)."),
    ] = None,
    reject_non_visible_rights: Annotated[
        bool | None,
        typer.Option(
            "--reject-non-visible-rights",
            help="Reject transitions needing non-visible castling/ep setup.",
        ),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output Parquet path for candidates."),
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
    """Retrieve ranked candidate next puzzles for a source puzzle."""
    _configure_logging(verbose)
    settings, repo = _open_analysis_repo(config)
    try:
        row = _fetch_puzzle_row(repo, puzzle_id)
        index = load_or_build_index(settings, repo, version=settings.similarity_version)
        backend_obj = get_backend(backend or settings.similarity_backend)
        score_weights, cost_weights = _similarity_weights(settings)
        reject = (
            settings.similarity_reject_non_visible_rights
            if reject_non_visible_rights is None
            else reject_non_visible_rights
        )
        candidates, stats = retrieve_with_index(
            settings=settings,
            index=index,
            source_puzzle_id=puzzle_id,
            source_row=row,
            end_policy_name=end_policy,
            backend=backend_obj,
            score_weights=score_weights,
            cost_weights=cost_weights,
            symmetries=settings.similarity_symmetries,
            rating_min=rating_min,
            rating_max=rating_max,
            max_rating_deviation=settings.similarity_max_rating_deviation,
            min_popularity=settings.similarity_min_popularity,
            piece_count_tolerance=settings.similarity_piece_count_tolerance,
            max_candidates_scanned=settings.similarity_max_candidates_scanned,
            shortlist_multiplier=settings.similarity_shortlist_multiplier,
            top_k=top or settings.similarity_default_top_k,
            reject_non_visible_rights=reject,
            exclude_unusual_material=settings.similarity_exclude_unusual_material,
        )
        console.print(
            f"[bold]Index:[/bold] {stats.index_rows} puzzles | "
            f"[bold]Filtered:[/bold] {stats.filtered_rows} | "
            f"[bold]Shortlist:[/bold] {stats.shortlisted_ids} | "
            f"[bold]Scored:[/bold] {stats.scored_candidates}"
        )
        console.print(
            f"[bold]Scan:[/bold] rows={stats.scan.rows_scanned} "
            f"pairs={stats.scan.pairs_materialised} "
            f"exact-hits={stats.scan.exact_lookup_hits} "
            f"truncated={stats.scan.truncated} ({stats.elapsed_seconds:.2f}s)"
        )
        _print_candidate_table(candidates)
        if output is not None:
            destination = output.resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            records = [candidate_to_record(candidate) for candidate in candidates]
            with pq.ParquetWriter(destination, CANDIDATE_SCHEMA, compression="zstd") as writer:
                writer.write_table(pa.Table.from_pylist(records, schema=CANDIDATE_SCHEMA))
            console.print(f"[green]Persisted[/green] {len(records)} candidates to {destination}")
    finally:
        repo.close()


@similarity_app.command("explain")
def similarity_explain_cmd(
    from_puzzle_id: Annotated[str, typer.Option("--from", help="Source puzzle id.")],
    to_puzzle_id: Annotated[str, typer.Option("--to", help="Next puzzle id.")],
    end_policy: Annotated[
        str,
        typer.Option("--end-policy", help="Stage-end policy of the source puzzle."),
    ] = "transition_ready",
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Path to YAML configuration file."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Explain the structural cost of one source -> next puzzle pair."""
    _configure_logging(verbose)
    settings, repo = _open_analysis_repo(config)
    try:
        row = _fetch_puzzle_row(repo, from_puzzle_id)
        endpoints, endpoint_source = resolve_accepted_endpoints(settings, row, end_policy)
        index = load_or_build_index(settings, repo, version=settings.similarity_version)
        policy = get_policy(end_policy)
        score_weights, cost_weights = _similarity_weights(settings)
        candidate = explain_pair(
            source_puzzle_id=from_puzzle_id,
            source_start_counts=_start_counts(row),
            endpoints=endpoints,
            index=index,
            next_puzzle_id=to_puzzle_id,
            end_policy_key=policy.key(),
            endpoint_source=endpoint_source,
            score_weights=score_weights,
            cost_weights=cost_weights,
            symmetries=settings.similarity_symmetries,
            reject_non_visible_rights=settings.similarity_reject_non_visible_rights,
        )
        if candidate is None:
            raise typer.BadParameter(
                f"puzzle {to_puzzle_id!r} is not present in the similarity index"
            )
        console.print(JSON(json.dumps(candidate.to_dict(), sort_keys=True, indent=2)))
    finally:
        repo.close()


@transition_app.command("synthesize")
def transition_synthesize_cmd(
    from_puzzle_id: Annotated[str, typer.Option("--from", help="Source puzzle id.")],
    to_puzzle_id: Annotated[str, typer.Option("--to", help="Next puzzle id.")],
    end_policy: Annotated[
        str,
        typer.Option("--end-policy", help="Stage-end policy of the source puzzle."),
    ] = "transition_ready",
    max_actions: Annotated[
        int | None,
        typer.Option("--max-actions", min=1, help="Bounded action search limit."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Write the best transition.json here."),
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
    """Synthesise and verify a transition from a source puzzle to a next
    puzzle (bounded search over 1..N actions; no LLM involved)."""
    _configure_logging(verbose)
    settings, repo = _open_analysis_repo(config)
    try:
        source_row = _fetch_puzzle_row(repo, from_puzzle_id)
        next_row = _fetch_puzzle_row(repo, to_puzzle_id)
        policy = get_policy(end_policy)
        context = build_context_from_row(source_row, policy)
        next_start = CanonicalPosition.from_board(chess.Board(str(next_row["start_fen"])))
        transition_config = settings.transition_config
        if max_actions is not None:
            transition_config = dataclasses.replace(
                transition_config, default_max_actions=max_actions
            )
        results = synthesize((context,), next_start, policy=policy, config=transition_config)
        if not results:
            console.print("[yellow]No verified transition found within the budget.[/yellow]")
            return
        for index, candidate in enumerate(results, start=1):
            console.print(f"[bold]Candidate {index}[/bold]")
            console.print(f"  instruction: {candidate.rendered}")
            console.print(
                f"  passed={candidate.verification.passed} "
                f"actions={len(candidate.program.actions)} "
                f"clue_bits={candidate.verification.clue.total_clue_bits:.2f} "
                f"absolute={_absolute_disclosures(candidate)}"
            )
            if candidate.verification.issues:
                console.print("  issues: " + " | ".join(candidate.verification.issues))
        if output is not None:
            best = results[0]
            document = build_transition_document(
                source_puzzle_id=from_puzzle_id,
                next_puzzle_id=to_puzzle_id,
                contexts=(context,),
                next_start=next_start,
                program=best.program,
                guard=best.guard,
                verification=best.verification,
                config=transition_config,
            )
            write_transition_document(output.resolve(), document)
            console.print(f"[green]Wrote[/green] {output.resolve()}")
    finally:
        repo.close()


@transition_app.command("verify")
def transition_verify_cmd(
    transition_path: Annotated[Path, typer.Argument(help="Path to a transition.json artifact.")],
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Path to YAML configuration file."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Reload a persisted transition AST and re-execute / re-verify it."""
    _configure_logging(verbose)
    artifact = load_transition_document(transition_path.resolve())
    if artifact is None:
        raise typer.BadParameter(f"cannot load transition document: {transition_path}")
    verification = artifact.re_verify()
    console.print(
        f"[bold]Transition:[/bold] {artifact.source_puzzle_id} -> {artifact.next_puzzle_id}"
    )
    console.print(f"[bold]Instruction:[/bold] {artifact.rendered}")
    console.print(f"[bold]Verified:[/bold] {verification.passed}")
    for issue in verification.issues:
        console.print(f"  [red]- {issue}[/red]")
    console.print(
        f"[bold]Clue report:[/bold] "
        f"total={verification.clue.total_clue_bits:.2f} bits, "
        f"absolute={_report_absolute(verification)}, "
        f"distinct_next={verification.clue.distinct_next_positions_across_plausible}"
    )
    if not verification.passed:
        raise typer.Exit(code=1)


@transition_app.command("explain-leakage")
def transition_explain_leakage_cmd(
    transition_path: Annotated[Path, typer.Argument(help="Path to a transition.json artifact.")],
    config: Annotated[
        Path | None,
        typer.Option("--config", help="Path to YAML configuration file."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Explain the full clue / leakage report of a persisted transition."""
    _configure_logging(verbose)
    artifact = load_transition_document(transition_path.resolve())
    if artifact is None:
        raise typer.BadParameter(f"cannot load transition document: {transition_path}")
    verification = artifact.re_verify()
    console.print(
        JSON(
            json.dumps(
                {
                    "source_puzzle_id": artifact.source_puzzle_id,
                    "next_puzzle_id": artifact.next_puzzle_id,
                    "rendered": artifact.rendered,
                    "clue_report": verification.clue.to_dict(),
                    "verification": verification.to_dict(),
                },
                sort_keys=True,
                indent=2,
            )
        )
    )


@sequence_app.command("generate")
def sequence_generate_cmd(
    config: Annotated[
        Path,
        typer.Option("--config", help="Path to a YAML sequence request file."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Where to write sequence.json."),
    ] = None,
    register: Annotated[
        bool,
        typer.Option(
            "--register/--no-register",
            help="Record the product in the local registry.",
        ),
    ] = True,
    config_file: Annotated[
        Path | None,
        typer.Option("--config-file", help="Path to YAML configuration file."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Generate a validated puzzle sequence from a request file."""
    _configure_logging(verbose)
    from chess_chain.sequence.generate import SequenceGenerationError, generate_sequence
    from chess_chain.sequence.registry import SequenceRegistry
    from chess_chain.sequence.request import request_from_yaml

    settings, repo = _open_analysis_repo(config_file)
    request = request_from_yaml(config)
    registry = SequenceRegistry.load(settings.sequence_dir / "registry.json")
    try:
        result = generate_sequence(
            request,
            settings,
            repo,
            registry=registry,
            register=register,
        )
    except SequenceGenerationError as exc:
        console.print(f"[red]Generation failed:[/red] {exc.message}")
        if exc.report is not None:
            console.print(JSON(json.dumps(exc.report.to_dict())))
        raise typer.Exit(1) from None
    destination = (output or settings.sequence_dir / f"{request.product_id}.json").resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    console.print(
        f"[bold]Generated[/bold] {request.product_id}: "
        f"{' -> '.join(result.puzzle_ids())} (valid={result.validation.valid}, "
        f"fingerprint={result.fingerprint[:12]})"
    )
    console.print(f"wrote {destination}")


@sequence_app.command("validate")
def sequence_validate_cmd(
    sequence_path: Annotated[Path, typer.Argument(help="sequence.json path.")],
    config_file: Annotated[
        Path | None,
        typer.Option("--config-file", help="Path to YAML configuration file."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Re-validate a persisted sequence document (re-executes transitions)."""
    _configure_logging(verbose)
    from chess_chain.sequence.graph import NeighbourGraph
    from chess_chain.sequence.records import load_sequence
    from chess_chain.sequence.validate import validate_sequence

    settings, repo = _open_analysis_repo(config_file)
    result = load_sequence(sequence_path)
    if result is None:
        raise typer.BadParameter(f"cannot load sequence document: {sequence_path}")
    graph = NeighbourGraph(
        settings,
        repo,
        transition_config=dataclasses.replace(
            settings.transition_config,
            default_max_actions=result.request.max_transition_actions,
        ),
    )
    report = validate_sequence(result, graph, result.request)
    console.print(
        JSON(
            json.dumps(
                {
                    "product_id": result.product_id,
                    "fingerprint": result.fingerprint,
                    "valid": report.valid,
                    "issues": list(report.issues),
                },
                sort_keys=True,
                indent=2,
            )
        )
    )
    if not report.valid:
        raise typer.Exit(1)


@sequence_app.command("explain")
def sequence_explain_cmd(
    sequence_path: Annotated[Path, typer.Argument(help="sequence.json path.")],
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Print the player-facing representation of a persisted sequence."""
    _configure_logging(verbose)
    from chess_chain.sequence.records import load_sequence

    result = load_sequence(sequence_path)
    if result is None:
        raise typer.BadParameter(f"cannot load sequence document: {sequence_path}")
    data = result.to_dict()
    console.print(JSON(json.dumps(data["player_facing"], sort_keys=True, indent=2)))
    console.print("[dim]Private manufacturing/QA data is intentionally not shown here.[/dim]")


@sequence_app.command("reproduce")
def sequence_reproduce_cmd(
    product_id: Annotated[str, typer.Option("--product-id", help="Registered product id.")],
    config_file: Annotated[
        Path | None,
        typer.Option("--config-file", help="Path to YAML configuration file."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Regenerate a registered product from its stored request and compare."""
    _configure_logging(verbose)
    from chess_chain.sequence.generate import SequenceGenerationError, generate_sequence
    from chess_chain.sequence.registry import SequenceRegistry
    from chess_chain.sequence.request import request_from_dict

    settings, repo = _open_analysis_repo(config_file)
    registry = SequenceRegistry.load(settings.sequence_dir / "registry.json")
    entry = registry.entries.get(product_id)
    if entry is None:
        raise typer.BadParameter(f"product {product_id!r} is not registered")
    request = request_from_dict(cast(dict[str, Any], entry["request"]))
    request = dataclasses.replace(request, product_id=product_id)
    try:
        result = generate_sequence(request, settings, repo, registry=registry, register=False)
    except SequenceGenerationError as exc:
        console.print(f"[red]Reproduction failed:[/red] {exc.message}")
        raise typer.Exit(1) from None
    matches = result.fingerprint == str(entry["fingerprint"])
    console.print(
        f"fingerprint={result.fingerprint[:16]} "
        f"expected={str(entry['fingerprint'])[:16]} "
        f"match={matches}"
    )
    if not matches:
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Stage 7: provisional rating estimate
# ---------------------------------------------------------------------------


@rating_app.command("estimate-simple")
def rating_estimate_simple_cmd(
    sequence_path: Annotated[Path, typer.Argument(help="sequence.json path.")],
    write_back: Annotated[
        bool,
        typer.Option(
            "--write-back/--no-write-back",
            help="Store the computed estimate in the sequence document.",
        ),
    ] = True,
) -> None:
    """Compute/refresh the provisional source-rating average placeholder."""
    from chess_chain.product.render import resolve_estimate, write_sequence_with_estimate
    from chess_chain.sequence.rating import MissingRatingError
    from chess_chain.sequence.records import load_sequence, write_sequence

    sequence = load_sequence(sequence_path)
    if sequence is None:
        raise typer.BadParameter(f"cannot load sequence document: {sequence_path}")
    try:
        estimate = resolve_estimate(sequence)
    except MissingRatingError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    console.print(JSON(json.dumps(estimate.to_dict(), indent=2, sort_keys=True)))
    console.print(
        f"[yellow]Caveat:[/yellow] {estimate.warnings[0] if estimate.warnings else '(none)'}"
    )
    if write_back:
        updated = write_sequence_with_estimate(sequence, estimate)
        write_sequence(updated, sequence_path)
        console.print(f"wrote estimate into {sequence_path}")


def _load_product(sequence_path: Path) -> Any:
    from chess_chain.sequence.records import load_sequence

    sequence = load_sequence(sequence_path)
    if sequence is None:
        raise typer.BadParameter(f"cannot load sequence document: {sequence_path}")
    return sequence


def _print_product_summary(artifacts: Any) -> None:
    console.print("[bold]Player-safe[/bold]:")
    console.print(f"  {artifacts.player_instructions}")
    console.print("[bold]Private manufacturing/developer material[/bold]:")
    console.print(f"  {artifacts.manufacturing_spec}")
    console.print(f"  {artifacts.qa_report}")
    console.print(f"  {artifacts.review_html}")
    console.print(f"  {artifacts.review_notes}")


@product_app.command("render")
def product_render_cmd(
    sequence_path: Annotated[Path, typer.Argument(help="sequence.json path.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory."),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite/--no-overwrite", help="Replace existing files."),
    ] = True,
    config_file: Annotated[
        Path | None,
        typer.Option("--config-file", help="Path to YAML configuration file."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Render all product artifacts from a sequence document."""
    from chess_chain.product.render import (
        render_product,
        resolve_estimate,
        write_sequence_with_estimate,
    )
    from chess_chain.sequence.records import write_sequence

    _configure_logging(verbose)
    settings, _ = _open_analysis_repo(config_file)
    sequence = _load_product(sequence_path)
    try:
        estimate = resolve_estimate(sequence)
    except Exception as exc:
        console.print(f"[red]Rating estimate failed: {exc}[/red]")
        raise typer.Exit(1) from None
    stored = sequence.difficulty_estimates.get(estimate.key)
    if stored is None or tuple(int(r) for r in stored.get("stage_ratings", [])) != tuple(
        sequence.stage_ratings()
    ):
        updated = write_sequence_with_estimate(sequence, estimate)
        write_sequence(updated, sequence_path)
        sequence = updated
        console.print(f"[yellow]Stored provisional estimate in {sequence_path}[/yellow]")
    artifacts = render_product(
        sequence, settings, output_dir=output, estimate=estimate, overwrite=overwrite
    )
    console.print(f"[bold]Rendered product[/bold] {sequence.product_id} to {artifacts.output_dir}")
    _print_product_summary(artifacts)


@product_app.command("validate")
def product_validate_cmd(
    sequence_path: Annotated[Path, typer.Argument(help="sequence.json path.")],
    explain_failure: Annotated[
        bool,
        typer.Option("--explain-failure", help="Print the exact failing location."),
    ] = False,
    config_file: Annotated[
        Path | None,
        typer.Option("--config-file", help="Path to YAML configuration file."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Independently reconstruct and re-verify the product specification."""
    from chess_chain.product.validate import validate_product

    _configure_logging(verbose)
    settings, repo = _open_analysis_repo(config_file)
    sequence = _load_product(sequence_path)
    report = validate_product(sequence, settings, repo=repo)
    console.print(
        JSON(
            json.dumps(
                {
                    "product_id": sequence.product_id,
                    "valid": report.valid,
                    "checks": report.checks,
                    "issues": [issue.to_dict() for issue in report.issues],
                },
                sort_keys=True,
                indent=2,
            )
        )
    )
    if explain_failure and not report.valid:
        rendered = report.explain_failure().replace("\n", "\n[/red]\n[red]")
        console.print("[red]" + rendered + "[/red]")
    if not report.valid:
        raise typer.Exit(1)


@product_app.command("review")
def product_review_cmd(
    sequence_path: Annotated[Path, typer.Argument(help="sequence.json path.")],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output directory."),
    ] = None,
    config_file: Annotated[
        Path | None,
        typer.Option("--config-file", help="Path to YAML configuration file."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable debug logging."),
    ] = False,
) -> None:
    """Render only the private visual review HTML (+ review-notes template)."""
    from chess_chain.product.render import (
        ARTIFACT_REVIEW_HTML,
        ARTIFACT_REVIEW_NOTES,
        resolve_estimate,
        write_sequence_with_estimate,
    )
    from chess_chain.product.review import render_review_html, render_review_notes
    from chess_chain.sequence.records import write_sequence

    _configure_logging(verbose)
    settings, _ = _open_analysis_repo(config_file)
    sequence = _load_product(sequence_path)
    estimate = resolve_estimate(sequence)
    stored = sequence.difficulty_estimates.get(estimate.key)
    if stored is None or tuple(int(r) for r in stored.get("stage_ratings", [])) != tuple(
        sequence.stage_ratings()
    ):
        updated = write_sequence_with_estimate(sequence, estimate)
        write_sequence(updated, sequence_path)
        sequence = updated
    directory = (output or settings.analysis_dir / "products" / sequence.product_id).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / ARTIFACT_REVIEW_HTML).write_text(
        render_review_html(sequence, settings, estimate), encoding="utf-8"
    )
    (directory / ARTIFACT_REVIEW_NOTES).write_text(render_review_notes(sequence), encoding="utf-8")
    console.print(f"[bold]Private review report:[/bold] {directory / ARTIFACT_REVIEW_HTML}")
    console.print(
        "[yellow]This is developer/manual-review material and may reveal every "
        "solution. Never ship it to players.[/yellow]"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
