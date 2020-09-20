#!/bin/bash

set -e

PLATFORM=("rocm")
FRAMEWORKS=("pytorch")
SEQ_LEN=(10 20 40 60 80 100 200 300 400 500)
BATCH_SIZE=(1 2 4 8 16 32 64 128)
MODEL=("bert-large-uncased")
N=100

PROF=
OUT_DIR=${1:-out}

for batch_size in ${BATCH_SIZE[*]}
do
	for seq_len in ${SEQ_LEN[*]}
	do
		for framework in ${FRAMEWORKS[*]}
		do
			for platform in ${PLATFORM[*]}
			do
				BENCH_LOG_DIR=$OUT_DIR/$platform/$MODEL_bs$BATCH_SIZE_seq$SEQ_LEN_$framework
				mkdir -p $BENCH_LOG_DIR
				BENCH_LOG=$BENCH_LOG_DIR/$MODEL.csv
				if [ "$platform" == "rocm" ]; then
					ROCBLAS_LAYER=2
					ROCBLAS_BENCH_LOG=$BENCH_LOG
				fi
				python src/benchmark.py --model $MODEL --seq_len=$seq_len --batch_size=$batch_size \
					--iters=$N --framework=$framework --platform=$platform
			done
		done
	done
done
