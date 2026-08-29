"""Raw source puzzle rows from the Lichess puzzle export."""

from __future__ import annotations

from dataclasses import dataclass

from chess_chain.domain.errors import IngestRejectError

#: Official header of the Lichess puzzle export (including the line terminator).
PUZZLE_HEADER = (
    "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,"
    "Themes,GameUrl,OpeningTags,DailyDate\n"
)


@dataclass(frozen=True)
class SourcePuzzle:
    """Immutable raw source fields for one puzzle row."""

    puzzle_id: str
    fen: str
    moves: tuple[str, ...]
    rating: int
    rating_deviation: int
    popularity: int
    nb_plays: int
    themes: tuple[str, ...]
    game_url: str
    opening_tags: tuple[str, ...]
    daily_date: str


def source_from_fields(fields: list[str], line_number: int) -> SourcePuzzle:
    """Parse one CSV field list into a SourcePuzzle or raise IngestRejectError."""
    if len(fields) < 10:
        raise IngestRejectError(
            f"line {line_number}: expected at least 10 fields, got {len(fields)}"
        )
    try:
        rating = int(fields[3])
        rating_deviation = int(fields[4])
        popularity = int(fields[5])
        nb_plays = int(fields[6])
    except ValueError as exc:
        raise IngestRejectError(f"line {line_number}: non-numeric rating field") from exc

    moves = tuple(move for move in fields[2].split(" ") if move)
    themes = tuple(fields[7].split())
    opening_tags = tuple(fields[8].split())
    daily_date = fields[10] if len(fields) > 10 else ""

    return SourcePuzzle(
        puzzle_id=fields[0],
        fen=fields[1],
        moves=moves,
        rating=rating,
        rating_deviation=rating_deviation,
        popularity=popularity,
        nb_plays=nb_plays,
        themes=themes,
        game_url=fields[9],
        opening_tags=opening_tags,
        daily_date=daily_date,
    )
