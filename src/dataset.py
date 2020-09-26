#!/usr/bin/env python3.6


import torch


class DatasetMLM():
    def __init__(self, bs, seq_len, vocab_size):
        input_ids = torch.randint(low=0,
            high=vocab_size - 1,
            size=(bs, seq_len),
            dtype=torch.long
            ).cuda()
        labels = torch.randint(low=0,
            high=vocab_size - 1,
            size=(bs, seq_len),
            dtype=torch.long
            ).cuda()
        next_sentence_label = torch.zeros(bs, dtype=torch.long).cuda()
        self.inputs = {'input_ids': input_ids, 'labels': labels, 'next_sentence_label': next_sentence_label}
#        if fp16:
#            for k, v in self.inputs.items():
#                self.inputs[k] = v.half()
        
    def get(self):
        return self.inputs


class DatasetLM(DatasetMLM):
    def __init__(self, bs, seq_len, vocab_size):
        input_ids = torch.randint(low=0,
            high=vocab_size - 1,
            size=(bs, seq_len),
            dtype=torch.long
            ).cuda()
        labels = torch.randint(low=0,
            high=vocab_size - 1,
            size=(bs, seq_len),
            dtype=torch.long
            ).cuda()
        self.inputs = {'input_ids': input_ids, 'labels': labels}
#        if fp16:
#            for k, v in self.inputs.items():
#                self.inputs[k] = v.half()

