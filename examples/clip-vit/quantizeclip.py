from PIL import Image
import requests

import torch
import onnxruntime
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
from datasets import load_dataset
from scipy.special import softmax

model_name = "openai/clip-vit-base-patch32"

class ClipDataReader(CalibrationDataReader):
    def __init__(self, tensor_type='np') -> None:
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.currentIndex = 0
        self.datasize = 1
        self.tensor_type = tensor_type

    def get_next(self):
        if self.currentIndex >= self.datasize:
            return None

        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = self.processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors=self.tensor_type, padding=True)

        model_inputs = {
            'input_ids':   inputs['input_ids'],
            'pixel_values':  inputs['pixel_values'],
            'attention_mask': inputs['attention_mask'],
        }

        if self.tensor_type == 'np':
            model_inputs['input_ids'] = model_inputs['input_ids'].astype(np.int64)
            model_inputs['attention_mask'] = model_inputs['attention_mask'].astype(np.int64)

        self.currentIndex += 1

        return model_inputs

    def rewind(self):
        self.currentIndex = 0

def quantize_model():
    pt_model = CLIPModel.from_pretrained(model_name)
    pt_model.eval()

    sample_inputs = ClipDataReader("pt").get_next()
    # pt_inputs = {k: torch.tensor(v) for k, v in sample_inputs.items()}

    input_model_path = "./clip-vit-base-patch32.onnx"
    torch.onnx.export(
        pt_model,
        (sample_inputs['input_ids'], sample_inputs['pixel_values'], sample_inputs['attention_mask']),
        f=input_model_path,
        input_names=['input_ids', 'pixel_values', 'attention_mask'],
        output_names=['logits_per_image', 'logits_per_text', 'text_embeds', 'image_embeds'],
    )

    preproc_model_path = "./clip-vit-base-patch32_qnn_pre.onnx"
    modified = qnn_preprocess_model(
        input_model_path,
        preproc_model_path,
        # fuse_layernorm=True,
        # save_as_external_data=True,
        # external_data_location="./clip-vit-base-patch32_qnn_pre.onnx_data",
        # all_tensors_to_one_file=True
    )

    model_to_quantize = preproc_model_path if modified else input_model_path

    quant_pre_model_path = "./clip-vit-base-patch32_quant_pre.onnx"
    quant_pre_process(
        input_model=model_to_quantize,
        output_model_path=quant_pre_model_path,
        # save_as_external_data=True,
        # external_data_location="./clip-vit-base-patch32_quant_pre.onnx_data",
        # all_tensors_to_one_file=True,
    )

    calib_reader = ClipDataReader()

    quant_config = get_qnn_qdq_config(
        model_input=quant_pre_model_path,
        calibration_data_reader=calib_reader,
        activation_type=QuantType.QUInt16,
        weight_type=QuantType.QUInt8
    )

    quant_config.op_types_to_quantize = ['MatMul', 'Gemm', 'LayerNormalization', 'Sigmoid', 'Gelu']  # ok

    print("Quantizing model")
    quantize(
        model_input=quant_pre_model_path,
        model_output="./clip-vit-base-patch32_quantized.onnx",
        quant_config=quant_config,
    )
    print("Model quantized")

def test_model():
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    processor = CLIPProcessor.from_pretrained(model_name)
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="np", padding=True)

    model_inputs = {
        'input_ids':   inputs['input_ids'].astype(np.int64),
        'pixel_values':  inputs['pixel_values'],
        'attention_mask': inputs['attention_mask'].astype(np.int64),

    }

    options = onnxruntime.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    session = onnxruntime.InferenceSession(
        # "./clip-vit-base-patch32_quantized.onnx",
                                           r"examples\clip-vit\models\clip-vit-base-patch32\model\model.onnx",
                                        # sess_options=options,
                                        providers=["QNNExecutionProvider"],
                                        provider_options=[{"backend_path": "QnnHtp.dll"}])

    outputs = session.run(["logits_per_image", "logits_per_text"], model_inputs)
    probs = softmax(outputs[0], axis=1)
    print(probs)

quantize_model()
test_model()
