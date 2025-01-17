from pathlib import Path
import json 

import torch
import onnxruntime
import numpy as np
from transformers import AutoTokenizer, BertTokenizer, BertModel, BertForQuestionAnswering
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
from datasets import load_dataset
import datetime

model_name = "google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")
model_dir = Path("./")  # TODO: change this to the directory where the models are saved
max_length = tokenizer.model_max_length

def preprocess(question, context, fixed_length=None):
    if fixed_length is not None:
        assert fixed_length > 0
        fixed_length = min(fixed_length, tokenizer.model_max_length)
        params = {
            'padding': 'max_length',
            'truncation': True,
            'max_length': fixed_length,
            'return_tensors': 'np'
        }
    else:
        params = {
            'return_tensors': 'np'
        }
        
    encoded_input = tokenizer(
        question,
        context,
        **params
    )
    tokens = tokenizer.convert_ids_to_tokens(list(encoded_input.input_ids.reshape(-1)))
    return (encoded_input.input_ids, encoded_input.attention_mask, encoded_input.token_type_ids, tokens)

def postprocess(tokens, start, end):
    results = {}
    answer_start = np.argmax(start)
    answer_end = np.argmax(end)
    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
        results['answer'] = answer.capitalize()
    else:
        results['error'] = "I am unable to find the answer to this question. Can you please ask another question?"
    return results

class SquadV2DataReader(CalibrationDataReader):
    def __init__(self, split='train', datasize=0, fixed_length=None) -> None:
        self.fixed_length = fixed_length
        self.dataset = load_dataset("rajpurkar/squad_v2", split=split)
        self.currentIndex = 0
        self.datasize = len(self.dataset) if datasize <= 0 else min(datasize, len(self.dataset))

    def get_next(self):
        if self.currentIndex >= self.datasize:
            return None
        input_ids, attention_mask, token_type_ids, tokens = preprocess(
            self.dataset[self.currentIndex]["question"],
            self.dataset[self.currentIndex]["context"],
            fixed_length=self.fixed_length,
        )
        model_inputs = {
            'input_ids':   np.array(input_ids, dtype=np.int64),
            'attention_mask':  np.array(attention_mask, dtype=np.int64),
            'token_type_ids': np.array(token_type_ids, dtype=np.int64),
        }
        self.currentIndex += 1
        print(f'{self.currentIndex}/{self.datasize}')
        
        return model_inputs

    def rewind(self):
        self.currentIndex = 0

def quantize_model():
    pt_model = BertForQuestionAnswering.from_pretrained(model_name)
    pt_model.eval()

    sample_inputs = SquadV2DataReader(fixed_length=max_length).get_next()
    pt_inputs = {k: torch.tensor(v) for k, v in sample_inputs.items()}

    torch.onnx.export(
        pt_model,
        (pt_inputs['input_ids'], pt_inputs['attention_mask'], pt_inputs['token_type_ids']),
        f='./bert.onnx',
        do_constant_folding=True, 
        input_names=['input_ids', 'attention_mask', 'token_type_ids'],
        output_names=['start_logits', 'end_logits'],
    )

    modified = qnn_preprocess_model(
        str(model_dir / "bert.onnx"),
        str(model_dir / "bert_qpm.onnx"),
        fuse_layernorm=True,
        save_as_external_data=True,
        external_data_location="bert_qpm.onnx_data",
        all_tensors_to_one_file=True
    )
    assert modified, "Model was not modified"

    quant_pre_process(
        input_model=str(model_dir / "bert_qpm.onnx"),
        output_model_path=str(model_dir / "bert_qpm_pre.onnx"),
        save_as_external_data=True,
        external_data_location=r"bert_qpm_pre.onnx_data",
        all_tensors_to_one_file=True,
    )

    print('Quantizing...')
    calib_reader = SquadV2DataReader(datasize=200, split='train', fixed_length=max_length)

    quant_config = get_qnn_qdq_config(
        # model_input=str(model_dir / "bert_fixed_pre_tile.onnx"),
        model_input=str(model_dir / "bert_qpm_pre.onnx"),
        calibration_data_reader=calib_reader,
        activation_type=QuantType.QUInt16,
        weight_type=QuantType.QUInt8,
        # weight_type=QuantType.QUInt16,  # test if QUInt16 weights can maintain chat capability
        # nodes_to_quantize=['Add'],
    )

    # quant_config.op_types_to_quantize = ['MatMul', 'Gemm', 'LayerNormalization', 'Sigmoid'] # ok
    quant_config.op_types_to_quantize = ['MatMul', 'Gemm', 'LayerNormalization', 'Sigmoid', 'Gelu']  # ok
    # quant_config.op_types_to_quantize = ['MatMul', 'Gemm', 'LayerNormalization', 'Sigmoid', 'Gelu', 'Add']  # Add bad 
    # quant_config.op_types_to_quantize = ['MatMul', 'Gemm', 'LayerNormalization', 'Sigmoid', 'Gelu', 'Softmax'] # Bad

    quantize(
        model_input=str(model_dir / "bert_qpm_pre.onnx"),
        model_output=str(model_dir / "bert_quantized.onnx"),
        quant_config=quant_config,
    )

def test_model():
    strbeg = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(f"Testing quantized model on CPU with sample inputs...{strbeg}")

    inputs = "{\"question\": \"What is Dolly Parton's middle name?\", \"context\": \"Dolly Rebecca Parton is an American singer-songwriter\"}"
    inputs = json.loads(inputs)

    input_ids, attention_mask, token_type_ids, tokens = preprocess(inputs["question"], inputs["context"], fixed_length=max_length)

    model_inputs = {
        'input_ids':   np.array(input_ids, dtype=np.int64),
        'attention_mask':  np.array(attention_mask, dtype=np.int64),
        'token_type_ids': np.array(token_type_ids, dtype=np.int64),
    }

    # session = onnxruntime.InferenceSession(str(model_dir / "bert_quantized.onnx"))
    
    options = onnxruntime.SessionOptions()
    options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

    session = onnxruntime.InferenceSession(str(model_dir / "bert_quantized.onnx"),
                                        # sess_options=options,
                                        providers=["QNNExecutionProvider"],
                                        provider_options=[{"backend_path": "QnnHtp.dll"}])

    strbeg = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(f"session run begin...{strbeg}")
    outputs = session.run(['start_logits', 'end_logits'], model_inputs)
    strbeg = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(f"session run end...{strbeg}")
    results = postprocess(tokens, outputs[0], outputs[1])
    print(results)  # Should be "Rebecca"
    strend = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(strend)

test_model()
