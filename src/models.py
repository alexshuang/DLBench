#!/usr/bin/env python3.6


import torch
import transformers
from transformers import AdamW
from torch.optim import Optimizer
from apex import amp


model_mappings = {
        'bert': transformers.BertForPreTraining,
        }


config_mappings = {
        'bert': transformers.BertConfig,
        }


class ModelMLM():
    def __init__(self, model_id, fp16=False):
        mtype = model_id.split('-')[0]
        config = config_mappings[mtype].from_pretrained(model_id)
        model = model_mappings[mtype](config).cuda()
        if fp16:
            no_decay = ["bias", "LayerNorm.weight"]
            optim = AdamW([p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)])
        else:
            optim = AdamW([p for n, p in model.named_parameters()])

        if fp16:
            model, optim = amp.initialize(model, optim, opt_level='O1')

        self.fp16, self.model, self.config, self.optim = fp16, model, config, optim

    def train(self, **kwargs):
        outputs = self.model(**kwargs)
        loss = outputs[0]

        if self.fp16:
            with amp.scale_loss(loss, self.optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        max_grad = 20
        if self.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), max_grad)
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad)
        self.optim.step()
        self.model.zero_grad()

