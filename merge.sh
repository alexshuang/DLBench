#!/bin/bash

set -e

LOG_BENCH_DIR=${1:-out/log/rocm}

start_time=$(date +%s)

echo "Merging ROCm rocblas bench log ..."
LOGS=`find $LOG_BENCH_DIR -name \*.csv`
TOTAL_LOG=$LOG_BENCH_DIR/rocblas_bench.csv
for l in $LOGS; do
    cat $l >> $TOTAL_LOG
done
sort -k2n $TOTAL_LOG | uniq > /tmp/rocblas_bench.csv
cp /tmp/rocblas_bench.csv $TOTAL_LOG

end_time=$(date +%s)
cost_time=$[ $end_time-$start_time ]

echo "Benchmark done in $(($cost_time/60))min $(($cost_time%60))s"

