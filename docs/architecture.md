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

Castling is a single UCI that moves two physical pieces (king *and* rook);
`derive_core` and the transition context replays track both, so the castled
rook keeps its stage identity, is recorded as moved, and can appear in the
final-checking set (e.g. rook to f8/f1/d8/d1 delivering the check).

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


## Stage 4 (puzzle similarity organiser)

The similarity organiser retrieves promising puzzle pairs `(i, j)` where the
solved final position of puzzle `i` can be transformed into the presented start
of puzzle `j` with short, low-leakage physical instructions. It does **not**
generate natural-language transitions and does not build full sequences yet: it
produces ranked candidate pairs and explains their structural costs. The
domain invariant is preserved: for every accepted endpoint `e` of stage `i`,
`transition(e) = presented_start_j`. Intermediate accepted endpoints may have
different occupancy and piece placement (no intermediate mechanical check);
they are never rejected for that reason, but are scored separately and
aggregated by best / mean / worst.

### Layered position distance

Instead of a generic embedding, the distance between one accepted endpoint and
one transformed next start is layered:

1. **Occupancy distance**: `d_occ = popcount(final_occupancy XOR start_occupancy)`.
2. **Piece-placement distance**: popcount over the 12 colour/type bitboards
   (`d_placement`); a swapped pair contributes 4 (two bits per bitboard).
3. **Material / inventory compatibility**: on-board pieces plus the tray
   (everything the stage started with that is no longer on the endpoint board =
   `max(0, start_counts - endpoint_counts)` per type, which covers captures and
   set-aside promoted pawns). Normal-set limits, promotions and surplus pieces
   are reported as annotations; `inventory_valid` is per endpoint.
4. **Minimum-cost physical piece assignment**: per (colour, type) sub-problems
   solved exactly with SciPy `linear_sum_assignment` (the Hungarian algorithm is
   not re-implemented). Unmatched on-board pieces are removals; tray pieces are
   additions; an addition onto a vacated square is a replacement; matched
   on-board pairs that exchange squares are swaps.
5. **Relative displacement cost**: file/rank deltas for every matched pair,
   staying on board, plus replacement/add/remove accounting.
6. **Whole-board transformations**: identity, file reflection, rank reflection
   (colours and side to move exchange) and 180-degree rotation (colours and
   side to move exchange). Every transformed candidate is reconstructed as a
   python-chess board and validated with `Board.is_valid()`; a symmetry that
   does not preserve a legal standard-chess position is skipped for that
   candidate. In particular, file reflection maps a king e1 -> d1 and a 180
   rotation maps e1 -> d8, so castling rights cannot survive those transforms;
   rank reflection swaps colours and keeps rights (white KQ -> black kq).

Castling rights and the en-passant square are carried through comparisons: two
visually equal boards with different rights are *not* the same exact next
puzzle (`exact_position_match` uses the full `PositionFingerprint`). Endpoints
that would require the player instructions to establish non-visible rights are
flagged (`castling_annotation_required` / `ep_annotation_required`), and
product configuration (`similarity.reject_non_visible_rights`) can reject
them.


- **Source:** [Lichess open database](https://database.lichess.org/#puzzles)
  puzzle export (`lichess_db_puzzle.csv.zst`, CC0). Fields:
  `PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags,DailyDate`.
- **Acquisition:** `chess-chain db download` (stage 1) with SHA-256 sidecar
  manifest; raw files are immutable after success.
- **Derivation:** `chess-chain db ingest` (stage 2) records
  `transform_version` in the dataset manifest; bumping it invalidates old
  outputs.

## Stage 5 (deterministic transition language and synthesiser)

Stage 5 turns ranked candidate pairs into concrete transition programs. A typed
transition AST is the *only* semantic authority: the executor executes and
verifies the AST, the renderer formats it from templates, and the persisted
artifact stores the AST plus machine-verifiable before/after FENs. **No LLM
invents executable transition instructions.** All instructions are visible to
the player before solving; there are no hidden codes or staged reveals.

### Grammar

Selectors resolve to physical pieces (stage-local `StagePieceId`s); a selector
that matches zero pieces is `missing_selector` and one matching several is
`ambiguous_selector` (ties are never silently chosen).

| Selector | Meaning |
|----------|---------|
| `piece_began_on_square(X)` | the piece that began the stage on square X |
| `occupant_of_square(X)` | the current occupant of square X |
| `unique_piece(colour, type)` | the unique piece of that colour/type |
| `last_moved_piece` / `piece_made_ply(n)` | pieces from the intended history |
| `first_captured_piece` / `last_captured_piece` / `captured_piece_began_on_square(X)` | tray pieces |
| `checking_piece` / `mating_piece` / `checked_king` | check state |
| `nearest_to(anchor, colour, farthest)` | equidistant ties are rejected |
| `in_direction(anchor, direction)` | the first piece along a ray |

Destinations: `relative_offset(piece, dx, dy)` (no wrap), `square_of(piece)`,
`vacated_by(piece)` / `ply_origin(n)` (squares vacated by tracked pieces or
move events), `relative_to_anchor(anchor, dx, dy)`, and `absolute_square(X)` as
a permitted but penalised fallback.

Actions: `move`, `swap`, `remove`, `replace` (captured/off-board piece placed
on an occupied square), `add` (off-board piece to an empty square),
`transform(symmetry)` (whole-board, reconstructed and validated with
python-chess), `establish_rights(castling_mask, ep_square)` (instruction
metadata - rights are never inferable from static placement) and
`conditional(branches)` (only when every branch is fully verifiable and each
accepted endpoint takes exactly one branch to the same next start).

The executor tracks each physical piece through the working board, rejects
duplicate use, collisions, off-board destinations and inventory violations,
and voids castling rights in standard-chess fashion when a king or rook leaves
its home square. The final position must match the next start's full
`PositionFingerprint` (bitboards, occupancy, turn, castling, en-passant). The
side to move is preserved by the transition; whole-board transforms that
exchange colours also exchange the turn.


### Orientation

The renderer never says "left" or "up". Offsets are written with board-global
file and rank names: "two files towards h", "three ranks towards rank 8". The
player-facing global rule ("continue choosing the strongest move ... until
checkmate or the stated transition condition becomes uniquely applicable") is a
configurable string, not a hard-coded assumption.

### Stage-boundary guards

A typed `StageBoundaryGuard` is separate from the transition actions even when
both are rendered as one instruction. For every intermediate `StageEndPolicy`
the same rule applies: the guard must be **false or non-unique** on the
presented start, every proper intended prefix and the configured plausible
wrong states, and **true and uniquely executable** on every accepted endpoint.

- `CHECKMATE` - python-chess checkmate; the guard records that the player
  learns mate is the goal (`goal_leakage`).
- `TRANSITION_READY` - the guard *is* the transition becoming executable and
  producing the exact next start.
- `EXPLICIT_PREDICATE` - the printed condition is compiled to a formal
  predicate and verified; disclosures count named captured pieces, occupied
  squares, checks and promotions.
- `FIXED_PLIES` - the stated length is verified and the revealed line length is
  recorded.

An intermediate puzzle needs no occupancy invariance: different accepted
endpoint occupancies are allowed when the same generic program (or explicit
verified branches) maps them all to the exact same next start.

### Leakage and the epistemic-anchor rule

For a no-wrap relative vector `(dx, dy)` on an 8x8 board the number of possible
origins is `N = (8 - |dx|) * (8 - |dy|)` and the geometric estimate is
`log2(64 / N)` bits (`(6, 6)` -> 4 origins -> 4 bits). When a plausible
endpoint distribution is available the conditional leakage
`H(before) - H(after)` is used instead, where `before` is the anchor piece's
candidate final squares and `after` keeps only the lines in which the
instruction is on-board, unambiguous and executable; the report labels whether
weighted entropy or log-cardinality was used.

The relevant quantity is uncertainty *before solving*, never whether a piece
moved. An unmoved queen can be a strong anchor (many plausible lines would move
it); a moved piece with one forced destination has near-zero uncertainty.

For each candidate program `verify` checks exact correctness on every accepted
endpoint, invalidity on initial/prefix/wrong states (recording whether each
state is executable, ambiguous, off-board, colliding or inventory-invalid),
the number of distinct next positions produced across plausible endpoints
(all collapsing to the exact next start is rejected when configured), and the
configurable minimum anchor uncertainty and minimum independent anchors.
Absolute-square disclosures, occupied-square disclosures and the guard
disclosures are folded into one `total_clue_bits` figure rather than rating the
move instructions in isolation.

### Synthesis and persistence

`transition synthesize --from X --to Y` derives the `EndpointContext` (presented
start, exact endpoint, tracked identities, full move history, captures in
order, last-moved and checking/mating pieces, rights/ep, policy), then searches
boundedly over 1..N actions: whole-board transforms first, then short relative
operations derived from the similarity piece-assignment plans, with
replacements/additions/removals/swaps and absolute placements as fallbacks.
Candidates are executed and verified before ranking (correctness, invalidity on
non-accepted states, low reconstructability, low conditional leakage, low
absolute disclosure, few actions).

`transition verify transition.json` reloads the AST and re-executes it, so a
renderer change can never alter semantics; `transition explain-leakage` prints
the full clue report. The persisted document stores the schema, source/next
puzzle ids, policy key, every accepted endpoint's presented FEN and solution
line, the exact next-start FEN, the program AST, the guard, the rendered text
and the verification + clue report.


## Stage 6 (complete puzzle sequences)

Stage 6 composes the repository, analysis, similarity and transition modules
into complete unique puzzle sequences. **Puzzles are nodes and synthesised
transitions are directed edges; no global all-pairs graph over the corpus is
materialised.** Neighbours are retrieved on demand per node.

### Versioned request

`SequenceRequest` (schema `sequence@1`) carries: stage count; target
source-rating range or provisional adjusted-rating target; permitted rating
spread; minimum popularity and play count; maximum rating deviation; allowed
and forbidden themes; solver-uniqueness strictness; opponent-forcedness policy;
allowed/preferred intermediate `StageEndPolicy` values; the final-only
occupancy-collision status and scope; maximum transition actions; maximum
leakage; maximum absolute-square disclosures; the physical inventory profile;
the exact-journey requirement; the final checkmate requirement (always on by
default); the random seed; product/sequence id; exclusions and already-used
puzzle ids; plus the search knobs (beam width, per-node candidate cap,
backtrack budget, validated-finalist alternatives, finalists checked).

### Hard constraints

Every source line replays legally; every solver move meets the configured
uniqueness policy; every opponent move meets the forcedness policy (or the
accepted branches are explicitly supported); every intermediate stage has a
verified visible ending; every transition maps **every** accepted intermediate
endpoint to the same next presented start (intermediate endpoints do not have
to share occupancy); transitions and guards stay within leakage/complexity
limits; all stages are achievable with the configured inventory; no puzzle is
repeated within a sequence; exclusions/already-used ids are respected; the last
stored solution ends in python-chess-confirmed checkmate; **only the final
puzzle must have occupancy invariance** across its accepted lines; and **only
the final puzzle** must meet the requested cached
`collision_free_within_scope` status - never renamed to an unbounded proof.

### On-demand neighbour generation

`NeighbourGraph.edges` queries the similarity index for candidate next starts in
the requested rating band (rating spread, quality gates, theme filters,
inventory), runs exact piece assignment (`retrieve`), attempts transition
synthesis, and caches successful *and failed* edge attempts by the complete
version/config fingerprint (in-memory plus a disk cache under
`analysis/sequence/edge_cache`). Every edge carries decomposed costs
(occupancy distance, action count, leakage, absolute disclosures, exact
journey) and rejection reasons.

### Search

The search strategy sits behind a `SequenceSearcher` protocol so an
OR-Tools/CP-SAT solver can be added later. `BeamSearchSearcher` is a bounded,
reproducible depth-first expansion over ranked per-path neighbour edges: each
expanded path retains its top-`width` ranked children (the beam), and a dead
end - or a failed final selection - pops the next sibling at the deepest
unexhausted level, bounded by the backtrack budget. Ties are broken with a hash
of `(seed, path)` so the same seed reproduces the same output while different
seeds pick different valid sequences when choices exist. Transparent
lexicographic scoring orders the optimisation preferences: soundness and
forcedness, recognisability, requested difficulty, exact continuous journey,
occupancy-continuous journey, minimal actions, low leakage, low absolute
disclosure, varied themes, minimal setup burden.

### Lazy final-candidate selection

Collision analysis never runs for every puzzle and never for intermediate
nodes. Finalists are ranked first with cheap/deep cached features (rating
proximity, popularity, transition cost, journey, theme variety); then, in rank
order, each finalist's collision request is looked up in the persistent
final-collision cache (`analysis/collision_cache`) under the complete
scope/configuration key; a miss runs `FinalOccupancyCollisionAnalyzer` for that
candidate only and stores the result. Collisions are rejected, and under strict
policy inconclusive results are rejected too; a configurable number of
validated finalist alternatives is retained.

### Outputs

The canonical `sequence@1` document contains the request, the stable sequence
fingerprint, per-stage records (accepted lines, endpoint FENs, the verified
transition edge with its AST, guard disclosures, final collision report), the
machine validation report, the search report (chosen nodes/edges and rejected
alternatives), the provisional player-facing representation (first setup,
side to move per stage when required, all visible transitions - no hidden
codes), and private manufacturing/QA data (intended lines, accepted branches,
endpoint FENs, intermediate guards, the single final mechanical occupancy
target and its bounded collision report). A local JSON registry records
product ids, fingerprints and consumed puzzle ids; `forbid_puzzle_reuse` is
configurable because enforcing it consumes the corpus much faster than merely
requiring unique fingerprints. CLI: `sequence generate --config request.yaml`,
`sequence validate sequence.json`, `sequence explain sequence.json`,
`sequence reproduce --product-id ...`.


## Stage 7 (final integration: rating placeholder, product rendering, validation)

### Provisional difficulty boundary

`sequence/rating.py` defines the `RatingEstimator` protocol and a versioned
`DifficultyEstimate` (`method`, `scale`, `provisional`, `label`, `value`,
`count`, `minimum`, `maximum`, `stage_ratings`, `estimator_version`,
`warnings`). `SimpleAverageRatingEstimator` computes only the arithmetic mean
of the stage source ratings, in order; a missing/invalid source rating raises
`MissingRatingError` by default (no silent imputation). The label is always
**"Provisional source-rating average"**; no uncertainty interval is invented
and `value` is a rounded integer. Results are stored in the `sequence@1`
record under a forward-compatible `method@estimator_version` key so a future
calibrated estimator can coexist without invalidating old records. The
placeholder is intentionally simplistic (it ignores the both-sides protocol,
visible transition clues, stage-end information and the multiplicative
difficulty of completing several stages) and is for manual product review
only.

### Product renderer

`product/render.py` materialises the canonical record into five strictly
separated artifacts:

- `player_instructions.md` - the only player-safe artifact: board orientation,
  initial setup, castling/en-passant statement, global rule, configured stage
  boundaries and every transition instruction. It never links to, embeds or
  names solution-bearing artifacts.
- `manufacturing_spec.json` - the canonical, rendering-independent product
  specification: product/generator versions, source puzzle ids + CC0
  provenance, first position, per-stage `StageEndPolicy` + wording, intended
  and accepted lines, endpoint FENs, transition ASTs + rendered text,
  next-start FENs, the single final mechanical occupancy mask, final checkmate
  proof, final-only bounded collision status/scope/witnesses, the provisional
  rating object with full metadata, deterministic seed + fingerprints.
- `qa_report.md` - hard-constraint pass/fail (from the stored machine
  validation; `product validate` re-checks independently), warnings,
  transition and stage-boundary leakage decompositions, per-ply forcedness,
  source ratings + provisional average, reproduce commands.
- `sequence_review.html` - a self-contained private developer report
  (python-chess SVG boards embedded with CSS into one file, Jinja2 template,
  no server needed). Intended replay one ply at a time (SAN + UCI + board),
  accepted branches clearly separated, endpoint boards, guard results,
  player-visible transition text beside the private AST summary, the board
  after the transition vs the next stage's start with an exact-match indicator,
  leakage decomposition, inventory warnings, and final-only checkmate /
  occupancy-mask / collision-status sections. Anchors per stage, overview
  table, product/fingerprint/seed/reproduce identifiers and a manual checklist
  textarea (not persisted).
- `review_notes.yaml` - a human review-aid template (status/notes/rejected
  stages/transitions/suggested changes) with a copy-ready `excluded_puzzle_ids`
  example; never read back by any command.

### Independent product validation

`product/validate.py` reconstructs the product from the specification and
re-verifies: initial setup legality, every accepted move, intermediate
stage-ending guards (guard re-verification on the source stage's accepted
endpoints), no intermediate occupancy-invariance requirement, final accepted
line occupancy invariance, final-only collision status against its declared
bounded scope/policy, transition execution from every accepted endpoint,
exact next-start positions, explicit handling of non-visible
castling/en-passant rights, physical inventory feasibility, final checkmate,
the sequence fingerprint, analysis/config version staleness and provisional
rating internal consistency. `--explain-failure` reports the exact stage, ply,
transition AST node or inventory item responsible.

### Configuration layering

Precedence (highest wins) is explicit and tested: CLI overrides
(`analyze deep --engine-path`, `--multipv`, `analyze variations --engine-path`,
`db ingest --output`) > environment variables (`CHESS_CHAIN_*`; for
machine-specific paths such as Stockfish) > user YAML request file
(`--config`) > checked-in defaults. `Settings.from_yaml` drops YAML-derived
values that have a matching environment variable so env vars really beat the
file (pydantic-settings init kwargs would otherwise win).

CLI: `rating estimate-simple sequence.json`, `product render|validate|review`.
`product render` materialises the placeholder into the record when absent or
stale. `sequence@1` records now also carry `difficulty_estimates` and
`versions`; `EdgeAttempt` records carry the persisted clue/guard/verification
summaries so the canonical spec is self-contained for rendering.


## Directory roles


| `src/chess_chain/domain/` | Typed domain structures |
| `src/chess_chain/chess/` | python-chess based replay/fingerprint logic |
| `src/chess_chain/analysis/` | Stage-3 tiers: engine, forcedness, variations, collision |
| `src/chess_chain/similarity/` | Stage-4 similarity organiser: symmetry, assignment, distance, index, backends, retrieval |
| `src/chess_chain/transitions/` | Stage-5 deterministic transition AST, executor, guards, leakage, synthesis, renderer, records |
| `src/chess_chain/sequence/` | Stage-6 sequence generation: versioned request, on-demand graph, beam search, lazy finalists, registry, validation, records |
| `src/chess_chain/sequence/rating.py` | Stage-7 provisional difficulty boundary: `RatingEstimator` protocol, `SimpleAverageRatingEstimator` |
| `src/chess_chain/product/` | Stage-7 product rendering (`player`, `spec`, `qa`, `review`), independent validation, review template |
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
| `similarity` | — | — | — | Puzzle similarity (stage 4: ranked candidate pairs) |
| `transitions` | — | — | — | Deterministic transition language + synthesiser (stage 5: typed AST, executor, guards, leakage, renderer) |
| `sequence` | — | — | — | Complete puzzle sequences (stage 6: versioned request, on-demand graph, beam search, lazy finalists, registry) |
| `rating` | — | — | — | Difficulty estimation |
| `product` | — | — | — | Stage 7: product rendering + independent validation (player instructions, manufacturing spec, QA report, review HTML, review notes) |
| `reporting` | — | Stats | Summaries | Export and reports |

## Configuration

`config/default.yaml` defines data paths, download/ingest settings (batch size,
row-group size, rating bucket width, transformation version) and reserved
analysis defaults. Paths are resolved relative to the project root unless
absolute.

### Retrieval pipeline (no all-pairs materialisation)

Never compare all 6M x 6M pairs. Per query:

1. filter candidate rows by rating band and product-quality constraints first
   (rating deviation, popularity, unusual-material exclusion);
2. group or partition the remaining rows by piece count and material signature,
   after a componentwise material-feasibility prefilter
   (`start_counts_j <= available` for every type);
3. query nearest occupancy masks within those groups (one query vector per
   group, hard-capped by `max_candidates_scanned`), with an exact
   occupancy-hash lookup for distance zero;
4. retain only a configurable top-K shortlist;
5. apply the exact per-endpoint piece-assignment costs to that shortlist and
   rank by the decomposable transition-complexity score.

Per-query work is linear in the scanned rows (each group is compared against
the single query mask), never quadratic. `ScanStats` reports
`groups_scanned`, `rows_scanned`, `pairs_materialised` (== rows_scanned),
`exact_lookup_hits` and `truncated`, proving boundedness.

### Backends

`SimilarityBackend` is a protocol:

- `exact` - NumPy vectorised `bitwise_count` baseline (default);
- `accelerated` - optional SciPy `cdist` `hamming` kernel over unpacked bits
  (compiled, cross-platform; SciPy is already a project dependency). Both
  return identical results; acceleration stays behind the interface and is
  never mandatory (no FAISS-style native packages).

### Candidate scoring

The `transition_complexity_score` is a weighted sum of named, explainable
components (all weights in `similarity.score_weights`): best/mean/worst
occupancy distance, best/mean/worst assignment cost, total swaps and
replacements, a non-identity symmetry penalty, a penalty when any endpoint
needs non-visible rights, an inventory-invalid penalty and a per-branch penalty
(one program must normalise every accepted endpoint). Lower is better; every
component is exposed so the number is never unexplained. `reasons_for_rejection`
records inventory infeasibility, assignment infeasibility and the
non-visible-rights product policy.

### Index persistence and versioning

`similarity build-index` writes `index.parquet` (schema `INDEX_SCHEMA`) plus
`index.manifest.json`. The manifest carries a version fingerprint over the
similarity algorithm version, the dataset transform version, the source
fingerprint, the dataset name and the row count, so stale indexes are detected
and transparently rebuilt by `query` / `explain`.

### CLI (`chess-chain similarity ...`)

- `similarity build-index [--where ...]`
- `similarity query --puzzle-id X [--rating-min ...] [--rating-max ...] [--top 50] [--end-policy ...] [--backend ...] [--output candidates.parquet]`
- `similarity explain --from X --to Y [--end-policy ...]`

Every expensive command reports the index/filter/shortlist sizes and the
bounded scan accounting.

### Expected complexity

Index build is O(N) row scans plus O(N) map construction (bounded memory). A
single query is O(filters + grouped_rows) vectorised 64-bit popcounts with
`grouped_rows <= max_candidates_scanned`; the shortlist receives the exact
piece-assignment analysis (small per-colour/type Hungarian problems). The
synthetic 20k-row benchmark fixture runs in well under a second on development
hardware. Full-corpus (multi-million row) wall-clock timings are not promised;
production runs should tune `max_candidates_scanned` and `piece_count_tolerance`.

## Dataset provenance


