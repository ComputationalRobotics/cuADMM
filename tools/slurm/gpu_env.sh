# Environment of the pinned configuration (tools/reference/pinned_config.md), sourced on the compute node.
# FASRC: modules cuda/12.9.1, gcc/12.2.0, cmake/3.31.6; SuiteSparse 7.10.1 / CHOLMOD 5.3.1 / OpenBLAS 0.3.34 from
# $HOME/cuadmm-env; CUADMM_THREADS BLAS threads (default 16; the CHOLMOD factorization and solves run on the CPU).
# The thread count is part of the pinned configuration: every run of one configuration uses the same value.
source /etc/profile.d/modules.sh 2>/dev/null
module load cuda/12.9.1-fasrc01 gcc/12.2.0-fasrc01 cmake/3.31.6-fasrc01 >/dev/null 2>&1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$HOME/cuadmm-env/lib
export OMP_NUM_THREADS=${CUADMM_THREADS:-16}
export OPENBLAS_NUM_THREADS=${CUADMM_THREADS:-16}
export CUADMM_REF_BUILDS=${CUADMM_REF_BUILDS:-$HOME/cuadmm-builds}
export CUADMM_BASELINES=${CUADMM_BASELINES:-$HOME/cuadmm-baselines}
export CUADMM_DATA=${CUADMM_DATA:-$HOME/cuadmm-data}
export PYTHON=${PYTHON:-$HOME/cuadmm-analysis-env/bin/python}
export PYTHONDONTWRITEBYTECODE=1
