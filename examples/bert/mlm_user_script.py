# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
import random
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizer
from olive.data.registry import Registry
import evaluate

class QuantizationDataset(Dataset):
    def __init__(self, model_name, start=0, end=100):
        assert 0 <= start < end
        self.start = start
        self.end = end
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        iterable_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        iterable_dataset=iterable_dataset.skip(self.start)
        self.it = iter(iterable_dataset)

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, idx):
        item = next(self.it)
        encoded_input = self.tokenizer(
                    item["text"],
                    padding="max_length",
                    max_length= 128,
                    truncation=True,
                    return_tensors='np')

        return {
                "input_ids": encoded_input['input_ids'].astype(np.int64),
                "attention_mask": encoded_input['attention_mask'].astype(np.int64),
                "token_type_ids": encoded_input['token_type_ids'].astype(np.int64)
            }

@Registry.register_dataset()
def quantization_dataset(**kwargs):
    return QuantizationDataset(**kwargs)

class MLMDataset(QuantizationDataset):
    def __getitem__(self, idx):
        item = next(self.it)
        # Replace random word with [MASK]
        words = item["text"].split()
        random_idx = random.randint(0, min(30, len(words)))
        words[random_idx] = "[MASK]"
        masked_text = " ".join(words)

        encoded_input = self.tokenizer(
                    masked_text,
                    padding="max_length",
                    max_length= 128,
                    truncation=True,
                    return_tensors='np')

        return {
                "input_ids": encoded_input['input_ids'].astype(np.int64),
                "attention_mask": encoded_input['attention_mask'].astype(np.int64),
                "token_type_ids": encoded_input['token_type_ids'].astype(np.int64)
            }, torch.Tensor([random_idx]).to(torch.int32)

@Registry.register_dataset()
def mlm_dataset(**kwargs):
    return MLMDataset(**kwargs)

def calc_perplexity(model_output, targets, model_name):
    # model_output[0]: preds, model_output[1]: logits
    # calculate metric
    # return metric value
    # model_outputs["input_ids"]

    probs = model_output.logits[torch.arange(model_output.logits.size(dim=0)), targets, :].softmax(dim=-1)
    _, predictions = probs.topk(5)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    for p in predictions:
        tokenizer.decode([p])

    perplexity = evaluate.load("perplexity", module_type="measurement")
    input_texts = ["Hello I'm a model model.", "Hello I'm a world model.", "Hello I'm a data model.", "Hello I'm a flight model.", "Hello I'm a business model." ]
    results = perplexity.compute(model_id='gpt2',
                                add_start_token=False,
                                data=input_texts)
    results["mean_perplexity"]
