from PIL import Image
import requests

from transformers import CLIPProcessor
import onnxruntime
import numpy as np

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="np", padding=True)

options = onnxruntime.SessionOptions()
session = onnxruntime.InferenceSession(r"C:\Users\zhangchao\project\Olive\examples\clip\models\clip-vit-base-patch32\model.onnx",
                                       providers=["QNNExecutionProvider"],
                                       provider_options=[{"backend_path": "QnnHtp.dll"}])
outputs = session.run(["logits"],
                     {
                        "input_ids": inputs['input_ids'].astype(np.int64),
                        "attention_mask": inputs['attention_mask'].astype(np.int64),
                        "pixel_values": inputs['pixel_values'].astype(np.int64)
                    })
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
print("Label probs:", probs)
