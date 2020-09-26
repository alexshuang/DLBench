#!/usr/bin/env python3.6


import argparse
import json
import os
import sys
import time
import tqdm
import torch
import contexttimer
from models import *
from dataset import *


def run_model(model, batch_size, seq_len, num_iter, framework_name):
    # warmup
    model()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    with contexttimer.Timer() as t:
        for it in range(num_iter):
            model()
    end.record()
    torch.cuda.synchronize()
    torch_elapsed = start.elapsed_time(end) / 1e3
    ips = num_iter / torch_elapsed

    time_consume = torch_elapsed
    '''
    print(json.dumps({
        "IPS": ips,
        "elapsed": time_consume,
        "avg_elapsed": time_consume / num_iter,
        "iter": num_iter,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "framework": framework_name,
        }))
    '''


def benchmark_pytorch(model:str, batch_size:list, seq_len:list, num_iter:int=1, fp16:bool=False):
    if not torch.cuda.is_available():
        print("cuda is not available for torch")
        return

    m = ModelMLM(model, fp16)

    bs_seq = [(bs, sl) for bs in batch_size for sl in seq_len]
    bar = tqdm.tqdm(bs_seq)
    for (bs, sl) in bar:
        bar.set_description(f"{model}")
        data = DatasetMLM(bs, sl, m.config.vocab_size)
        run_model(lambda: m.train(**data.get()), bs, sl, num_iter, 'pytorch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = "DL Benchmark"
    parser.add_argument("--model", type=str, help='model name')
    parser.add_argument("--seq_len", type=str, help='sequnce length list')
    parser.add_argument("--batch_size", type=str, help='batch size list')
    parser.add_argument("--num_iter", type=int, default=1, help='number of training iterations')
    parser.add_argument("--framework", type=str, help='framework name list')
    parser.add_argument("--platform", type=str, help='AMD or Nvidia list')
    parser.add_argument("--fp16", action="store_true", help='enable fp16')
    parser.add_argument("--bf16", action="store_true", help='enable bf16')
    args = parser.parse_args()

    batch_size = [eval(o) for o in args.batch_size.split('-')]
    seq_len = [eval(o) for o in args.seq_len.split('-')]
    fp16 = True if args.fp16 else False
    
    if args.framework == 'pytorch':
        benchmark_pytorch(args.model, batch_size, seq_len, args.num_iter, fp16)
    else:
        raise RuntimeError(f"Not supportted framework {args.framework}")
