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

class MobileNetDataset(Dataset):
    def __init__(self):
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = self.processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

        model_inputs = {
            'input_ids':   inputs['input_ids'],
            'pixel_values':  inputs['pixel_values'],
            'attention_mask': inputs['attention_mask'],
        }

        return model_inputs, "a photo of a cat"

@Registry.register_dataset()
def mobilenet_dataset(**kwargs):
    return MobileNetDataset()

@Registry.register_post_process()
def mobilenet_post_process(output):
    return output.argmax(axis=1)
