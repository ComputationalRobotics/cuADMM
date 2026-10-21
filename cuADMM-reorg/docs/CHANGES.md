# Behavior changes relative to cuADMM `main` (c23e67e)

Plan principle 4 and the developer prompt: no silent behavior change. Every change of results, option meaning or
default is listed here with the phase that introduced it and the new name or value that selects it. Structural
changes that reproduce the reference iterates bit for bit are not listed (they are verified by
`tools/compare_runs.py`; see `docs/PROGRESS.md`).

## Phase 0 (2026-10-21)

No behavior change: `cuADMM-reorg/` is a verbatim copy of `src/`, `include/`, `src/main.cu`, `MATLAB/` and the
used subset of `psd_projection` (b566590), built as its own project. The MEX target links the `cuadmm` library
instead of recompiling the sources.
