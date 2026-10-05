#!/bin/bash
# Captures the hardware/software environment of the allocation (run on the GPU node via srun).
OUT=$1; mkdir -p $OUT
{
  echo "date: $(date -Iseconds)"; echo "host: $(hostname)"
  echo "slurm job: $SLURM_JOB_ID partition: $(scontrol show job $SLURM_JOB_ID | grep -o 'Partition=[^ ]*')"
  echo "allocation command: ${ALLOC_CMD:-see logs/slurm_actions.log}"
  echo "srun: srun --jobid=$SLURM_JOB_ID --overlap -n 1 -c 16 --gres=gpu:nvidia_h200:1"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  scontrol show job $SLURM_JOB_ID | grep -E "TRES|NodeList|TimeLimit|StartTime"
} > $OUT/slurm.txt
nvidia-smi -L > $OUT/nvidia-smi-L.txt 2>&1
nvidia-smi -q > $OUT/nvidia-smi-q.txt 2>&1
nvidia-smi topo -m > $OUT/nvidia-smi-topo.txt 2>&1
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total,clocks.max.sm,clocks.max.mem,clocks.sm,power.limit,mig.mode.current,compute_mode --format=csv > $OUT/gpus.csv 2>&1
lscpu > $OUT/lscpu.txt; (numactl -H || lscpu -e) > $OUT/numa.txt 2>&1; free -g > $OUT/memory.txt; cat /proc/meminfo | head -5 >> $OUT/memory.txt
echo "affinity: $(taskset -pc $$)" > $OUT/affinity.txt; grep -E "Cpus_allowed_list|Mems_allowed_list" /proc/self/status >> $OUT/affinity.txt
{
  echo "gcc: $(gcc --version | head -1)"; echo "cmake: $(cmake --version | head -1)"; echo "nvcc: $(nvcc --version | tail -2 | tr '\n' ' ')"
  H=$(dirname $(which nvcc))/../include
  echo "cuda toolkit root: $(cd $(dirname $(which nvcc))/.. && pwd)"
  grep -h -E "#define CUBLAS_VER_(MAJOR|MINOR|PATCH)" $H/cublas_api.h | tr '\n' ' '; echo
  grep -h -E "#define CUSOLVER_VER_(MAJOR|MINOR|PATCH)" $H/cusolver_common.h | tr '\n' ' '; echo
  grep -h -E "#define CUSPARSE_VER_(MAJOR|MINOR|PATCH)" $H/cusparse.h | tr '\n' ' '; echo
  grep -h -E "#define CUDART_VERSION" $H/cuda_runtime_api.h
  grep -h -E "#define (CHOLMOD_(MAIN|SUB|SUBSUB)_VERSION|CHOLMOD_DATE)" $HOME/cuadmm-env/include/suitesparse/cholmod.h | tr '\n' ' '; echo
  grep -h -E "#define SUITESPARSE_(MAIN|SUB|SUBSUB)_VERSION" $HOME/cuadmm-env/include/suitesparse/SuiteSparse_config.h | tr '\n' ' '; echo
  echo "BLAS: $(readlink -f $HOME/cuadmm-env/lib/libblas.so) ; $(ls $HOME/cuadmm-env/lib | grep -i openblas | head -3 | tr '\n' ' ')"
  $HOME/cuadmm-env/bin/conda list 2>/dev/null | grep -E "openblas|suitesparse|cholmod|blas|lapack" || ls $HOME/cuadmm-env/conda-meta | grep -E "openblas|suitesparse|cholmod|blas|lapack"
  echo "modules: $(module -t list 2>&1 | tr '\n' ' ')"
} > $OUT/software.txt 2>&1
env | grep -E "OMP_|OPENBLAS|MKL_|CUDA|SLURM_CPUS|SLURM_JOB_GPUS|LD_LIBRARY_PATH|GOMP" | sort > $OUT/thread_env.txt
echo done
