#!/bin/bash

PLATFORM=("rocm")
FRAMEWORK=("pytorch")
MODEL_LIST=${1:-model_list.txt}
MODEL=`cat $MODEL_LIST`
#MODEL=("deepset/bert-large-uncased-whole-word-masking-squad2")
#SEQ_LEN="10-20-40-80-100-160-200-300-400-500"
#BATCH_SIZE="1-2-4-8-16-32-64"
#SEQ_LEN="large"
#BATCH_SIZE="large"
#SEQ_LEN="small"
#BATCH_SIZE="small"
SEQ_LEN="tiny"
BATCH_SIZE="tiny"
N=1

PROF=
OUT_DIR=${2:-out}
LOG_BENCH_DIR=$OUT_DIR/log
FP16="--fp16"
#NVPROF="sudo nvprof --track-memory-allocations off --concurrent-kernels off --print-gpu-trace --log-file bench.log --csv -f"

#rm -rf $LOG_BENCH_DIR

set -e

start_time=$(date +%s)

for platform in ${PLATFORM[*]}
do
    for framework in ${FRAMEWORK[*]}
    do
        for model in ${MODEL[*]}
        do
            if [ "$platform" == "rocm" ]; then
				BENCH_LOG=$LOG_BENCH_DIR/$platform/$framework/$model.csv
                BENCH_LOG_DIR=${BENCH_LOG%/*}
                mkdir -p $BENCH_LOG_DIR
                export ROCBLAS_LAYER=2
                export ROCBLAS_LOG_BENCH_PATH=$BENCH_LOG
            fi
            $PROF python src/benchmark.py --model $model --seq_len=$SEQ_LEN --batch_size=$BATCH_SIZE \
                --num_iter=$N --framework=$framework --platform=$platform $FP16 
        done
    done
done

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

