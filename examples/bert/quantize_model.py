# quantize_model.py

import numpy as np
import onnx
from onnxruntime.quantization import QuantType, quantize
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
import torch
import onnxruntime
from onnxruntime.quantization import CalibrationDataReader
from transformers import BertTokenizer, BertForMaskedLM

class DataReader(CalibrationDataReader):
    def __init__(self, model_path: str):
        self.enum_data = None

        # Use inference session to get input shape.
        session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        inputs = session.get_inputs()

        self.data_list = []

        # Generate 10 random float32 inputs
        # TODO: Load valid calibration input data for your model
        for _ in range(10):
            input_data = {inp.name : np.random.random(inp.shape).astype(np.float32) for inp in inputs}
            self.data_list.append(input_data)

        self.datasize = len(self.data_list)

    def get_next(self):
        if self.enum_data is None:
            self.enum_data = iter(
                self.data_list
            )
        return next(self.enum_data, None)

    def rewind(self):
        self.enum_data = None

def export_onnx_model(model, onnx_model_path):
    with torch.no_grad():
        inputs = {'input_ids':      torch.ones(1,128, dtype=torch.int64),
                    'attention_mask': torch.ones(1,128, dtype=torch.int64),
                    'token_type_ids': torch.ones(1,128, dtype=torch.int64)}

        torch.onnx.export(model,                                            # model being run
                    (inputs['input_ids'],                             # model input (or a tuple for multiple inputs)
                    inputs['attention_mask'],
                    inputs['token_type_ids']),                        # model input (or a tuple for multiple inputs)
                    onnx_model_path,                                  # where to save the model (can be a file or file-like object)
                    opset_version=17,                                 # the ONNX version to export the model to
                    # do_constant_folding=True,                         # whether to execute constant folding for optimization
                    input_names=['input_ids',                         # the model's input names
                                'attention_mask',
                                'token_type_ids'],
                    output_names=['logits'],                    # the model's output names
                )
# load model
model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")
export_onnx_model(model, "bert-base-multilingual-cased.onnx")

# input_model_path = "model.onnx"  # TODO: Replace with your actual model
# output_model_path = "model.qdq.onnx"  # Name of final quantized model
# my_data_reader = DataReader(input_model_path)

# # Pre-process the original float32 model.
# preproc_model_path = "model.preproc.onnx"
# model_changed = qnn_preprocess_model(input_model_path, preproc_model_path)
# model_to_quantize = preproc_model_path if model_changed else input_model_path

# # Generate a suitable quantization configuration for this model.
# # Note that we're choosing to use uint16 activations and uint8 weights.
# qnn_config = get_qnn_qdq_config(model_to_quantize,
#                                 my_data_reader,
#                                 activation_type=QuantType.QUInt16,  # uint16 activations
#                                 weight_type=QuantType.QUInt8)       # uint8 weights

# # Quantize the model.
# quantize(model_to_quantize, output_model_path, qnn_config)
