# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import BertTokenizer
from olive.data.registry import Registry

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

class MLMDataset(Dataset):
    def __init__(
        self,
        model_name,
        start=0,
        end=100
    ):
        assert 0 <= start < end
        self.start = start
        self.end = end
        self.model_name = model_name
        self.processor = CLIPProcessor.from_pretrained(self.model_name)  # max_length = 77 for input_ids
        self.length = self.end - self.start

        dataset = load_dataset(self.dataset_name, split=f"test[{self.start}:{self.end}]")
        text_inputs = self.processor(
            text=[" ".join(item["caption"]) for item in dataset],
            return_tensors="np",
            padding="max_length",
            truncation=True,
        )

        self.model_inputs = [
            {
                "input_ids": text_inputs["input_ids"].astype(np.int64),
                "attention_mask": text_inputs["attention_mask"].astype(np.int64),
            }
        ]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.model_inputs[idx], torch.Tensor([idx]).to(torch.int32)


@Registry.register_dataset()
def mlm_dataset(**kwargs):
    return MLMDataset(**kwargs)

@Registry.register_post_process()
def clip_post_process(output):
    return output["logits_per_image"].argmax(axis=-1)

def eval_perplexity(model, device, execution_providers):
    pass
