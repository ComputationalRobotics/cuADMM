# Job inspection at the change of plan, and the MOSEK cleanup (2026-09-29)

## Jobs at the change of plan

Inspected at 18:07:28 EDT with `date`, `squeue -u $USER`, `scontrol show job` and `scontrol write batch_script` for every job. Nothing was cancelled before this inspection.

| Job ID | Name | State | Purpose | Command/script | Output directory | Pending reason | Keep or cancel |
|---|---|---|---|---|---|---|---|
| 49301716 | cuadmm-cpu4 | PENDING | CPU placeholder for the CPU-only preparation: dataset conversion and checks, reference objectives, validator tests. It is not MOSEK and no solver is involved. | `sbatch -p shared -N 1 -c 16 --mem=64G -t 12:00:00 -J cuadmm-cpu4 --wrap "sleep 43000"`; the batch script is only `sleep 43000` | `~/cuadmm-tools/slurm_logs/cuadmm-cpu4-49301716.out` (the placeholder's own log) | Priority | Keep. It is not a MOSEK job, and it later started (18:33) and served the CPU work. |

No other job existed. Today's history (`sacct -S 2026-09-29T00:00`) held:

- 49082405: the fixed-σ campaign's H200 salloc, ended by TIMEOUT at 01:10;
- 49299681 and 49300430: two CPU sallocs, cancelled at 17:23 and 17:28 before they started. They were replaced by `sbatch` placeholders because `salloc` blocked during the maintenance.

**None of these jobs ran or requested MOSEK.**

Later CPU placeholder: 49308174 (`serial_requeue`, `sleep 21000`, submitted 18:10 because 49301716 was still pending).

## Cancellation

- No job was cancelled. The rule was "cancel only jobs whose sole purpose is MOSEK", and no such job existed.
- At 19:05 there was an attempt to cancel the idle, redundant CPU placeholder 49308174. It was not permitted in this session, so 49308174 runs until its time limit (00:14) and 49301716 until its limit (06:33).

## MOSEK outputs and the deletion manifest

**No MOSEK experiment of this study ran**, so there were no MOSEK outputs to delete.

Search, by file name only and without reading any contents:

- command: `find ~/cuadmm-worktrees/plato-cold/experiments ~/cuadmm-experiments ~/cuadmm-data ~/cuADMM/experiments ~/cuadmm-tools -iname '*mosek*'`;
- result: no file, apart from the public PLATO solver logs under `~/cuadmm-data/plato_kocvara/docs/sparse_logs/MOSEK/`.

Those 8 logs were downloaded from https://plato.asu.edu/ftp/sparse_logs/ as reference data. They are H. Mittelmann's runs, not ours. They are kept, because they are the source of the historical DIMACS accuracies and of MOSEK's published objectives (`data/historical_dimacs.csv`, `data/reference_objectives.csv`).

**Deletion manifest: 0 files deleted.**

The license file `$HOME/mosek/mosek.lic` was created by the user at 17:54. It was neither read, modified nor deleted: only its name, size and timestamp were listed once. `MOSEKLM_LICENSE_FILE` was not used.
