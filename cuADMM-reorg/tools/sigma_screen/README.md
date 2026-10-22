# tools/sigma_screen: the Phase-4b CPU screen of `balance_freeze`

The validated CPU replica of plain ADMM (`cuadmm_emu.py`, the sigma study's `emu-run/cuadmm_emu.py` v1.1 of
2026-10-19, sha256 `f213e9c6…`) extended with the `balance_freeze` rule (v1.2): `BalanceFreezeState` is a
transcription of `ResidualBalancingPenalty` with the freeze enabled (`src/admm/penalty_policy.cc`), selected with
`--rule balance_freeze[:key=value,...]` where the keys are the option names of `penalty.*` (`factor`, `step_schedule`,
`freeze_iteration`, `pingpong_window`, `pingpong_net`, `pingpong_min_iter`, `freeze_value`, `freeze_bias`,
`sigma_min`, `sigma_max`). The other rules (`old` = legacy_adaptive, `new` = residual_balancing, `fixed`, ...) are
untouched: the sigma study's suite `emu-run/test_emu_rules.py <this directory>` still passes.

Never run the replica on a login node: every command below goes through Slurm.

| file | purpose |
|---|---|
| `cuadmm_emu.py` | the replica (v1.2) |
| `test_balance_freeze.py` | tests of the v1.2 additions (`python -B test_balance_freeze.py` on a compute node) |
| `replay_policy.py` | drives `BalanceFreezeState` with the residuals of a reference history; writes the sequences that `test/penalty_policy_test.hpp` replays through the C++ policy (`test/data/golden/balance_freeze_cases_*.csv`) |
| `make_specs.py` | the run specifications of the screen (Stage 1, comparators, Stage 2, extra sigma_0 values) |
| `screen.sh`, `screen_task.sh` | the driver: one 1-core Slurm array task per unfinished run, checkpoints and resumes; pins a copy of the replica in `SCREEN_ROOT` |
| `analyze.py` | per-run metrics, per-setting tables and the Phase-4b gates (definitions in its docstring) |
| `specs/` | the spec files of the screen as run |

```
export SCREEN_ROOT=$HOME/cuadmm-experiments/2026-10-21_phase4b_screen
python -B make_specs.py stage1 specs/stage1.txt
python -B make_specs.py comparators specs/comparators.txt
./screen.sh specs/comparators.txt 24 00:15:00      # submit; call again to resume runs a task did not finish
./screen.sh specs/stage1.txt 40 00:15:00
srun -p <cpu partition> -c 1 python -B analyze.py specs/stage1.txt specs/comparators.txt
python -B make_specs.py stage2 specs/stage2.txt <best tag> <second tag>
./screen.sh specs/stage2.txt 40 00:15:00
```

Regenerating the C++ cross-check sequences (after a change of the rule in both implementations):

```
G=../../test/data/golden
srun -c 1 python -B replay_policy.py $G/PSD_Free_residual_balancing_s100_i6000_history.csv $G/balance_freeze_cases_PSD_Free.csv \
    "100@balance_freeze" "100@balance_freeze:freeze_iteration=3000,pingpong_min_iter=100000" ...
```
(the case list is in docs/PROGRESS.md, Phase 4b).
