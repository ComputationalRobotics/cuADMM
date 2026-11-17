# Build and run environment on a FASRC compute node: modules cuda/12.9.1, gcc/12.2.0, cmake/3.31.6; SuiteSparse
# 7.10.1 / CHOLMOD 5.3.1 / OpenBLAS 0.3.34 from $HOME/cuadmm-env; CUADMM_THREADS BLAS threads (default 16; the CHOLMOD
# factorization and solves run on the CPU). Results are bitwise reproducible for a fixed GPU model, toolchain and
# thread count.
source /etc/profile.d/modules.sh 2>/dev/null
module load cuda/12.9.1-fasrc01 gcc/12.2.0-fasrc01 cmake/3.31.6-fasrc01 >/dev/null 2>&1
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$HOME/cuadmm-env/lib
export OMP_NUM_THREADS=${CUADMM_THREADS:-16}
export OPENBLAS_NUM_THREADS=${CUADMM_THREADS:-16}
export CUADMM_DATA=${CUADMM_DATA:-$HOME/cuadmm-data}
export PYTHON=${PYTHON:-$HOME/cuadmm-analysis-env/bin/python}
export PYTHONDONTWRITEBYTECODE=1
