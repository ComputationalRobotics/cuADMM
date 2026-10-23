#!/bin/bash
# Runs the test runner of a build directory on the GPU node (the CLI tests call ./cuadmm_exe from the build directory).
# usage: run_tests.sh <build dir> <log file> [gtest filter]
set -u
HERE=$(cd "$(dirname "$0")/../.." && pwd)
source "$HERE/tools/slurm/gpu_env.sh"
BLD=$1; LOG=$2; FILTER=${3:-*}
cd "$BLD" || exit 2
export CUADMM_REGRESSION_DATA_DIR=$BLD/regression_data
echo "=== $(hostname) $(date) build $BLD filter $FILTER" > "$LOG"
nvidia-smi -L >> "$LOG"
./cuadmm_tests --gtest_filter="$FILTER" --gtest_color=no >> "$LOG" 2>&1
rc=$?
echo "=== exit $rc $(date)" >> "$LOG"
grep -E "^\[  (PASSED|FAILED) \]|tests ran|FAILED TEST" "$LOG" | tail -40
exit $rc
