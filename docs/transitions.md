# Transition language reference

The transition language is a typed, JSON-serialisable AST. The executor
(`src/chess_chain/transitions/executor.py`) is the only semantic authority: it
executes and verifies programs, and the renderer (`render.py`) turns the AST
into deterministic player-facing text. No LLM ever writes executable
instructions.

## Grammar (schema `transition@1`)

```
program     := action*
action      := move | swap | remove | replace | add | transform
             | establish_rights | conditional
move        := Move(piece=selector, destination=destination)
swap        := Swap(piece_a=selector, piece_b=selector)
remove      := Remove(piece=selector)
replace     := Replace(source=selector, destination=selector, with=piece_type, colour=colour)
add         := Add(source=selector, destination=destination, piece_type=piece_type, colour=colour)
transform   := Transform(symmetry=symmetry)           # whole-board, reconstructed + validated
establish_rights := EstablishRights(castling=rights, ep_square=square?)
conditional := Conditional(branches=(branch+))
branch      := Branch(condition=condition, actions=action*)
condition   := OnBoard(selector) | OffBoard(selector)
selector    := piece_began_on_square | occupant | unique_piece | last_moved | ply_piece
             | first_captured | last_captured | captured_began_on_square
             | checking_piece | mating_piece | checked_king
             | nearest_to(anchor) | farthest_from(anchor) | in_direction(anchor, dx, dy)
destination := relative_offset(piece, dx, dy) | square_of(piece)
             | vacated_by(piece) | ply_origin | relative_to_anchor(anchor, dx, dy)
             | absolute_square(square)
```

Selectors are physical identities, not just current positions: every piece
that begins a stage on the board is tracked through the stage so instructions
can say "the piece that began the stage on e8" regardless of where it moved.

## Semantics

- **Relative offsets** never wrap: `RelativeOffset(piece, dx, dy)` is a
  no-wrap vector, so off-board destinations are detected and reported, never
  silently clamped.
- **Ambiguity is an error.** A selector that resolves to zero or more than one
  physical piece fails with `missing`/`ambiguous`; ties are never broken
  silently.
- **Collisions and duplicate use are errors.** Two pieces may not land on one
  square; one physical piece may not be used twice in one program.
- **Inventory violations are errors.** Adds/replacements must come from the
  available physical inventory (a standard set by default).
- **Whole-board transforms** are reconstructed with python-chess and validated
  before use; only symmetries that produce a legal standard-chess position are
  accepted.
- **Castling / en-passant rights** are metadata instructions
  (`EstablishRights`), configurable via `transition.allow_establish_rights`
  and rejected when the product policy forbids them.

## Guards

A `StageBoundaryGuard` is separate from the transition actions even when both
are rendered as one instruction. Kinds: `CHECKMATE`, `TRANSITION_READY`
(the guard *is* the transition becoming executable and producing the exact next
start), `EXPLICIT_PREDICATE` (compiled to a formal predicate) and
`FIXED_PLIES`. Every intermediate guard must be false or non-unique on the
presented start, every proper intended prefix and the configured plausible
wrong states, and true and uniquely executable on every accepted endpoint.

## Verification

`verify_program` checks: exact correctness on every accepted endpoint;
invalidity on initial/prefix/wrong states (recording each state's status);
the number of distinct next positions produced across plausible endpoints;
minimum anchor uncertainty and minimum independent anchors; absolute-square
disclosures; and guard disclosures, folded into one `total_clue_bits` figure.

## Adding a selector or action

See `docs/contributing.md`.
