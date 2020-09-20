#!/usr/bin/env python3.6


import argparse
import json
import os
import sys
import torch
import transformers
import contexttimer


model_mappings = {
        'bert': transformers.BertForPreTraining,
        }


def run_model(model, batch_size, seq_len, framework_name):
    # warmup
    model()

    if use_cuda:
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
    print(json.dumps({
        "IPS": ips,
        "elapsed": time_consume,
        "avg_elapsed": time_consume / num_iter,
        "iter": num_iter,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "framework": framework_name,
        }))


def benchmark_pytorch(model:str, batch_size:int, seq_len:int, iters:int=1):
    if not torch.cuda.is_available():
        print("cuda is not available for torch")
        return

    model_name = model.split('-')[0]
    if model_name not in model_mappings:
        print(f"Not supportted model {model}")
        return

    m = model_mappings[model_name].from_pretrained(model).cuda()

    cfg = model.config
    input_ids = torch.randint(low=0,
        high=cfg.vocab_size - 1,
        size=(batch_size, seq_len),
        dtype=torch.long
        ).cuda()

    run_model(lambda: m(input_ids), batch_size, seq_len, 'pytorch')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = "DL Benchmark"
    parser.add_argument("--model", type=str, help='model name')
    #parser.add_argument("--model_id", type=str, help='model config id')
    parser.add_argument("--seq_len", type=int, help='sequnce length')
    parser.add_argument("--batch_size", type=int, help='batch size')
    parser.add_argument("--iters", type=int, default=1, help='iteration of benchmark')
    parser.add_argument("--framework", type=str, help='framework name')
    parser.add_argument("--platform", type=str, help='AMD or Nvidia')
    args = parser.parse_args()
    kwargs = vars(args)

    if args.framework == 'pytorch':
        benchmark_pytorch(args.model, args.batch_size, args.seq_len, args.iters)
    else:
        raise RuntimeError(f"Not supportted framework {args.framework}")

