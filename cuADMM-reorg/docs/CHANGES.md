# Behavior changes relative to cuADMM `main` (c23e67e)

Plan principle 4 and the developer prompt: no silent behavior change. Every change of results, option meaning or
default is listed here with the phase that introduced it and the new name or value that selects it. Structural
changes that reproduce the reference iterates bit for bit are not listed (they are verified by
`tools/compare_runs.py`; see `docs/PROGRESS.md`).

## Phase 0 (2026-10-21)

No behavior change: `cuADMM-reorg/` is a verbatim copy of `src/`, `include/`, `src/main.cu`, `MATLAB/` and the
used subset of `psd_projection` (b566590), built as its own project. The MEX target links the `cuadmm` library
instead of recompiling the sources.

## Pre-0 (2026-10-21): penalty policy names restored to the references

Inside `cuADMM-reorg/src/legacy/` (the copied `SDPSolver`), the CLI and the tests; plan Pre-0, decision Q1.

- `legacy_adaptive` is again the historical rule of commit 9bb4ba5, bit for bit: no guard (a change whenever the
  winner's count exceeds 1.35 times the other's), factor 2, sigma kept in [1e-6, 1e6], and the Monitor1 "chasing"
  correction after iteration 5000 every 25 iterations. On `main` since 2d8a755 the same name ran the October 13 rule,
  and since c23e67e with factor 1.2 (B18).
- `residual_balancing` (new name) is the October 13 rule of 2d8a755: guard `max(1, 1.35 other)`, factor
  `penalty.factor` (CLI `--penalty-factor`, default 2), sigma kept in [1e-3, 1e3], no Monitor1. The factor 1.2 of
  c23e67e is the option value `--penalty-factor 1.2`, never the meaning of a policy name.
- The default policy is `residual_balancing` (plan O1, until Phase 4b). A run of the old default (`main` at c23e67e,
  no `--sigma-policy`) therefore now runs the 2d8a755 rule with factor 2 instead of the 1.2-factor variant; the 1.2
  variant stays available as `--penalty-factor 1.2`.
- `fixed_sgs_then_legacy_admm` (the hybrid-only policy of `main`: fixed sigma during the sGS phase, then the
  plain-ADMM rule counted from the switch) is removed; the name is an error. It was neither reference rule, and the
  hybrid mode itself (`--sgs-iterations`, `--switch-admm`) stays available with every remaining policy. The two tests
  of that policy in `hybrid_switch_test.hpp` are removed with it.
- The bounds of the adaptive updates can be overridden (`--sigma-min`, `--sigma-max`; `penalty.sigma_min/max`).
  `sigma_0` is not clamped (as in both references; `balance_freeze`, Phase 4, clamps it as its pseudocode says).
- The summary JSON gains `sigma_policy_parameters` (factor, bounds, win ratio, guard, residual weight, Monitor1, sGS
  rule), `sigma_final_scaled`, `sigma_final_original`, `sigma0_original`, `build_project`, `build_commit` and
  `build_dirty` (the commit comes from a header generated at every build).
- The sigma log rule names: `admm_schedule`, `monitor1` (legacy_adaptive only), `sgs_rule`.
