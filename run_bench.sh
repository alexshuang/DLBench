#!/bin/bash

PLATFORM=("rocm")
FRAMEWORK=("pytorch")
#MODEL=("bert-large-uncased" "bert-base-uncased")
MODEL=`cat model_list.txt`
SEQ_LEN="10"
BATCH_SIZE="1"
N=1

PROF=
OUT_DIR=${1:-out}
FP16="--fp16"
#NVPROF="sudo nvprof --track-memory-allocations off --concurrent-kernels off --print-gpu-trace --log-file bench.log --csv -f"

set -e

for platform in ${PLATFORM[*]}
do
    for framework in ${FRAMEWORK[*]}
    do
        for model in ${MODEL[*]}
        do
            if [ "$platform" == "rocm" ]; then
				BENCH_LOG_DIR=$OUT_DIR/$platform/$framework
				mkdir -p $BENCH_LOG_DIR
				BENCH_LOG=$BENCH_LOG_DIR/$model.csv
                export ROCBLAS_LAYER=2
                export ROCBLAS_LOG_BENCH_PATH=$BENCH_LOG
            fi
            $PROF python src/benchmark.py --model $model --seq_len=$SEQ_LEN --batch_size=$BATCH_SIZE \
                --num_iter=$N --framework=$framework --platform=$platform $FP16
        done
    done
done

