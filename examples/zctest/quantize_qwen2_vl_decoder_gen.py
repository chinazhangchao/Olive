import argparse
from pathlib import Path
import re
import os
import pickle

import onnx
# from onnx import version_converter
import numpy as np
import datasets
from transformers import AutoTokenizer, AutoProcessor



import onnxruntime
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model

import utils

# TODO: get these from config json
model_name = "Qwen/Qwen2-VL-2B-Instruct"
num_layers = 28
num_heads = 2
hidden_size = 128
past_sequence_length = 0
input_dtype = np.float32

prompt_templates =  {
    "system": "<|im_start|>system\n{Content}<|im_end|>\n",
    "user": "<|im_start|>user\n{Content}<|im_end|>\n",
    "assistant": "<|im_start|>assistant\n{Content}<|im_end|>\n",
    "prompt": "<|im_start|>user\n{Content}<|im_end|>\n<|im_start|>assistant\n"
}

parser = argparse.ArgumentParser(description="Quantize Qwen vision encoder")
parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
parser.add_argument("--max_length", type=int, default=100, help="Max length of input sequence")
args = parser.parse_args()
model_dir = Path(args.model_dir)
max_length = args.max_length

print('Loading processor and tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name)

print('Loading calibration dataset...')
# TODO apply templates with tokenizer in data reader
# TODO: implement for more than one data samples
# TODO: add image tokens to dataset
class SlidinWindowDataReaderGen(CalibrationDataReader):
    def __init__(self, input_ids, pad_token_id, window_length, datasize=0) -> None:
        # TODO: precompute data
        self.input_ids = input_ids.reshape(-1)
        self.pad_token_id = pad_token_id
        self.window_length = window_length  # length of past KV
        self.index = 0
        self.datasize = len(self.input_ids) if datasize <= 0 else min(len(self.input_ids), datasize)
        self.embed_tokens_session = utils.make_cpu_session(
            str(model_dir / "embed_tokens_fp16.onnx"),
        )
        self.decoder_session = utils.make_cpu_session(
            str(model_dir / "decoder_model_merged_fp16.onnx"),
        )
        self.output_names = [f'present.{i}.key' for i in range(num_layers)] + [f'present.{i}.value' for i in range(num_layers)]

    def get_next(self):
        if self.index < self.datasize:
            # prepare data for KV
            input_ids = self.pad_token_id * np.ones((1, self.window_length), dtype=np.int64)
            attention_mask = np.zeros((1, self.window_length), dtype=np.int64)
            end_index = self.index
            start_index = max(0, end_index - self.window_length)
            position_ids = np.ones(max_length, dtype=np.int64)
            print(start_index, end_index)
            if end_index > 0:
                input_ids[0, -(end_index-start_index):] = self.input_ids[start_index:end_index]
                attention_mask[0, -(end_index-start_index):] = 1
                position_ids[-(end_index-start_index):] = np.arange(end_index-start_index, dtype=np.int64)
            position_ids = np.tile(position_ids.reshape(1, 1, -1), (3, 1, 1))  # m-rope
            inputs_embeds = self.embed_tokens_session.run(None, {'input_ids': input_ids.reshape(1, -1)})[0]

            inputs = {
                'attention_mask': attention_mask, 
                'inputs_embeds': inputs_embeds,
                'position_ids': position_ids,
            }
            for layer in range(num_layers):
                    inputs[f'past_key_values.{layer}.key'] = np.zeros((1, num_heads, past_sequence_length, hidden_size), dtype=input_dtype)
                    inputs[f'past_key_values.{layer}.value'] = np.zeros((1, num_heads, past_sequence_length, hidden_size), dtype=input_dtype)
            outputs = self.decoder_session.run(self.output_names, inputs)
            outputs_dict = dict(zip(self.output_names, outputs))

            # next token
            data_inputs_embeds = self.embed_tokens_session.run(None, {'input_ids': self.input_ids[end_index:end_index+1].astype(np.int64).reshape(1, -1)})[0]
            data_attention_mask = np.zeros((1, self.window_length + 1), dtype=np.int64)
            # data_attention_mask[0, -1] = 1
            data_attention_mask[0, -(end_index-start_index+1):] = 1 # last attention entry is for the new token
            data_position_ids = np.tile(np.array([end_index], dtype=np.int64).reshape(1, 1, -1), (3, 1, 1))
            data_inputs = {
                'attention_mask': data_attention_mask, 
                'inputs_embeds': data_inputs_embeds,
                'position_ids': data_position_ids,
            }
            for layer in range(num_layers):
                data_inputs[f'past_key_values.{layer}.key'] = outputs_dict[f'present.{layer}.key']
                data_inputs[f'past_key_values.{layer}.value'] = outputs_dict[f'present.{layer}.value']

            self.index += 1
            return data_inputs
        else:
            return None

    def rewind(self) -> None:
        self.index = 0

class CachedDataReader(CalibrationDataReader):
    def __init__(self, orig_reader, save_path, datasize=0) -> None:
        # precompute data and save to file
        self.index = 0
        if os.path.exists(save_path):
            with open(save_path, 'rb') as f:
                self.data = pickle.load(f)
            self.datasize = len(self.data) if datasize <= 0 else min(len(self.data), datasize)
            self.data = self.data[:self.datasize]
        else:
            self.data = []
            while True:
                inputs = orig_reader.get_next()
                if inputs is None or (datasize > 0 and len(self.data) >= datasize):
                    break
                self.data.append(inputs)
            with open(save_path, 'wb') as f:
                pickle.dump(self.data, f)
            self.datasize = len(self.data)

    def get_next(self):
        if self.index < self.datasize:
            inputs = self.data[self.index]
            self.index += 1
            return inputs
        else:
            return None
        
    def rewind(self) -> None:
        self.index = 0

role_pattern = re.compile('(\w+):(.+)', re.DOTALL)

role_map = {
    'Human': 'user',
    'Assistant': 'assistant'
}

def apply_template(messages):
        return ''.join(prompt_templates[role].format(Content=text) for role, text in messages)

def str_to_messages(x):
    chunks = x['text'].split('### ')
    messages = list()
    for chunk in chunks:
        m = role_pattern.match(chunk)
        if m:
            role = m.group(1)
            if role in role_map: # ignore unkown message types: "Title", "Table"
                role = role_map[role]
                text = m.group(2)
                messages.append((role, text))
    return messages

orig_dataset = datasets.load_dataset(
    "timdettmers/openassistant-guanaco",
    split="train",
)
data = apply_template(str_to_messages(orig_dataset[6]))
input_ids = tokenizer(data, return_tensors="np", padding='max_length', max_length=max_length)['input_ids']
data_reader = SlidinWindowDataReaderGen(input_ids, tokenizer.pad_token_id, max_length, datasize=200)

data_reader = CachedDataReader(data_reader, str(model_dir / 'cached_data.pkl'), datasize=200)

sample_inputs = SlidinWindowDataReaderGen(input_ids, tokenizer.pad_token_id, max_length).get_next()

print('Fixing shapes by running inference...')
utils.make_dynamic_shapes_fixed_by_sample(
     str(model_dir / "decoder_model_merged.onnx"),
     str(model_dir / "decoder_model_merged_fixed.onnx"),
    sample_inputs,
)

print('Running quant_pre_process...')
quant_pre_process(
    input_model=str(model_dir / "decoder_model_merged_fixed.onnx"),
    output_model_path=str(model_dir / "decoder_model_merged_pre.onnx"),
    save_as_external_data=True,
    external_data_location=r"decoder_model_merged_pre.onnx_data",
    # skip_symbolic_shape=True,
    # skip_onnx_shape=True,
    # skip_optimization=True,  # looking for optimized.onnx
    all_tensors_to_one_file=True,
)

print('Folding LayerNormalization...')
utils.fold_layernorms(
    input_model=str(model_dir / "decoder_model_merged_pre.onnx"),
    output_model=str(model_dir / "decoder_model_merged_pre_ln.onnx"),
    layernorm_keyword='norm',
    epsilon=1e-5,
)

print('Replacing Expand wiht Tile...')
utils.replace_expand_with_tile(
    input_model=str(model_dir / "decoder_model_merged_pre_ln.onnx"),
    output_model=str(model_dir / "decoder_model_merged_pre_ln_tile.onnx"),
)


# ----------------- BEGIN IGNORE -----------------
# print('Converting opset to 17...')
# model = onnx.load(str(model_dir / "decoder_model_merged_fixed.onnx"), load_external_data=False)
# # model.opset_import[0].version = 17
# converted_model = onnx.version_converter.convert_version(model, 17)
# onnx.save(
#     converted_model,
#     str(model_dir / "decoder_model_merged_17.onnx"),
#     save_as_external_data=True,
#     location="decoder_model_merged_17.onnx_data",
# )

# fuse_layernorm not working as the op pattern matched by this function may be different from
# print('Running qnn_preprocess_model...')
# modified = qnn_preprocess_model(
#     str(model_dir / "decoder_model_merged_17.onnx"),
#     str(model_dir / "decoder_model_merged_qpm.onnx"),
#     fuse_layernorm=True,
#     save_as_external_data=True,
#     external_data_location="decoder_model_merged_qpm.onnx_data",
#     # inputs_to_make_channel_last: list[str] | None = None,
#     # outputs_to_make_channel_last: list[str] | None = None,
# )
# print(modified)
# ----------------- END IGNORE -----------------

print('Quantizing...')
quant_config = get_qnn_qdq_config(
    model_input=str(model_dir / "decoder_model_merged_pre_ln_tile.onnx"),
    # calibration_data_reader=DataReader(calibration_dataset),
    calibration_data_reader=data_reader,
    activation_type=QuantType.QUInt16,
    weight_type=QuantType.QUInt8,
    # weight_type=QuantType.QUInt16,  # test if QUInt16 weights can maintain chat capability
    # nodes_to_quantize=['Add'],
)
# quant_config.op_types_to_quantize = ['MatMul', 'Gemm', 'LayerNormalization'] # ok
# quant_config.op_types_to_quantize = ['MatMul', 'Gemm', 'LayerNormalization', 'Mul'] # CPU ok; on NPU, meaningless text due to Mul, slower, NPU usage fragmented
quant_config.op_types_to_quantize = ['MatMul', 'Gemm', 'LayerNormalization', 'Sigmoid'] # result ok with Sigmoid
# quant_config.op_types_to_quantize = ['MatMul', 'Gemm', 'LayerNormalization', 'Sigmoid', 'Mul']  # ok but LM repeats more frequently
# quant_config.op_types_to_quantize = ['MatMul', 'Gemm', 'LayerNormalization', 'Sigmoid', 'Tile']  # QDQ Tile ok on CPU, meaningless on NPU; Tile leads to QNN graph with many EPContext, a bit slower
# TODO: QDQ reshape?

# quant_config.nodes_to_exclude = ['Tile', 'Reshape', 'Cast']  # meaningless output

quantize(
    model_input=str(model_dir / "decoder_model_merged_pre_ln_tile.onnx"),
    model_output=str(model_dir / "decoder_model_merged_quantized.onnx"),
    quant_config=quant_config,
)

print('Testing quantized model on CPU with sample inputs...')
session = onnxruntime.InferenceSession(str(model_dir / "decoder_model_merged_quantized.onnx"))
outputs = session.run(None, sample_inputs)

print('Done!')