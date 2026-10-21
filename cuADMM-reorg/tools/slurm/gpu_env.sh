# Environment of the pinned configuration (tools/reference/pinned_config.md), sourced on the compute node.
# FASRC: modules cuda/12.9.1, gcc/12.2.0, cmake/3.31.6; SuiteSparse 7.10.1 / CHOLMOD 5.3.1 / OpenBLAS 0.3.34 from
# $HOME/cuadmm-env; 16 BLAS threads (the CHOLMOD supernodal factorization and solves run on the CPU).
source /etc/profile.d/modules.sh 2>/dev/null
module load cuda/12.9.1-fasrc01 gcc/12.2.0-fasrc01 cmake/3.31.6-fasrc01 >/dev/null 2>&1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$HOME/cuadmm-env/lib
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export CUADMM_REF_BUILDS=${CUADMM_REF_BUILDS:-$HOME/cuadmm-builds}
export CUADMM_BASELINES=${CUADMM_BASELINES:-$HOME/cuadmm-baselines}
export CUADMM_DATA=${CUADMM_DATA:-$HOME/cuadmm-data}
export PYTHON=${PYTHON:-$HOME/cuadmm-analysis-env/bin/python}
export PYTHONDONTWRITEBYTECODE=1
