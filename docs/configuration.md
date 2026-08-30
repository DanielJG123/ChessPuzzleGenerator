# Configuration reference

Configuration is layered with explicit precedence (highest wins):

1. **CLI overrides** for a small number of common values
   (e.g. `analyze deep --engine-path`, `--multipv`; `analyze variations
   --engine-path`; `db ingest --output`, `--batch-size`).
2. **Environment variables** prefixed with `CHESS_CHAIN_`
   (e.g. `CHESS_CHAIN_ENGINE_PATH`, `CHESS_CHAIN_ENGINE_THREADS`,
   `CHESS_CHAIN_DATA_PROCESSED_DIR`). These are for machine-specific paths such
   as Stockfish and deliberately override user YAML.
3. **User YAML request files** (`--config path/to/default.yaml`) - checked in
   `config/default.yaml` or overridden per invocation.
4. **Checked-in defaults** (pydantic field defaults in `Settings` plus
   `config/default.yaml`).

`tests/test_product_stage7.py::test_config_precedence_cli_over_env_over_yaml_over_default`
locks this order.

## Top-level structure of `config/default.yaml`

| Section | Fields | Used by |
|---------|--------|---------|
| `paths` | `raw`, `interim`, `processed`, `samples` | all stages |
| `ingest` | `lichess_puzzle_url`, `lichess_puzzle_filename`, `download_chunk_size`, `download_space_margin_bytes`, `ingest_batch_size`, `parquet_row_group_rows`, `rating_bucket_width`, `transform_version` | `db download`, `db ingest` |
| `analysis.default_sample_rows` | `default_sample_rows` | `db sample` |
| `analysis.engine` | `path`, `threads`, `hash_mb`, `movetime_ms`, `depth`, `nodes`, `multipv` | `analyze deep` |
| `analysis.forcedness` | `version`, `unique_best_gap_cp`, `practically_forced_gap_cp`, `mate_value_cp`, `plausible_gap_cp` | `analyze deep`, `analyze variations` |
| `analysis.variations` | `default_max_plies`, `default_max_nodes`, `default_acceptance` | `analyze variations` |
| `analysis.collision` | `analyser_version`, `default_scope`, `default_max_depth`, `default_max_nodes` | `analyze final-collision`, sequence finalists |
| `similarity` | `version`, `backend`, `default_top_k`, `max_rating_deviation`, `min_popularity`, `piece_count_tolerance`, `max_candidates_scanned`, `shortlist_multiplier`, `reject_non_visible_rights`, `exclude_unusual_material`, `symmetries`, `cost_weights`, `score_weights` | `similarity build-index/query`, sequence neighbours |
| `transition` | `version`, `default_max_actions`, `plausible_depth`, `plausible_node_cap`, `min_anchor_uncertainty_bits`, `min_independent_anchors`, `allow_absolute_square_fallback`, `allow_establish_rights`, `reject_collapsing_reveal`, `global_rule_wording` | `transition synthesize/verify`, sequence transitions |

## SequenceRequest (per-product YAML)

A product request is a separate YAML document with top-level keys matching the
`SequenceRequest` dataclass fields in
`src/chess_chain/sequence/request.py`, for example:

```yaml
product_id: SEQ-1
num_stages: 3
rating_min: 1500
rating_max: 1700
target_rating: 1600
min_popularity: 10
allowed_intermediate_policies: [transition_ready]
collision_scope: up_to_plies
collision_max_depth: 12
max_transition_actions: 4
random_seed: 7
start_puzzle_id: null
excluded_puzzle_ids: []
already_used_puzzle_ids: []
```

Search knobs (`beam_width`, `per_node_candidates`, `backtrack_budget`,
`validated_finalist_alternatives`, `max_finalists_checked`) and the
accepted-line enumeration knobs (`acceptance`, `max_plies`, `max_nodes`,
`max_accepted_lines`) live in the same document. Unknown keys are ignored so a
future request version stays backward-compatible.
