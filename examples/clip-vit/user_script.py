# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from torch.utils.data import Dataset

from olive.data.registry import Registry
from transformers import CLIPProcessor
from PIL import Image
import requests
import numpy as np
import torch

class MobileNetDataset(Dataset):
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = self.processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="np", padding=True)

        model_inputs = {
            'input_ids':   inputs['input_ids'].astype(np.int64),
            'pixel_values':  inputs['pixel_values'],
            'attention_mask': inputs['attention_mask'].astype(np.int64),
        }

        return model_inputs, torch.Tensor([0]).to(torch.int32)

@Registry.register_dataset()
def mobilenet_dataset(**kwargs):
    return MobileNetDataset()

@Registry.register_post_process()
def mobilenet_post_process(output):
    return torch.Tensor([[output[0].argmax()]]).to(torch.int32)
