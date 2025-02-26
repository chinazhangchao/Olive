# run_qdq_model.py

import onnxruntime
import numpy as np
from transformers import BertTokenizer
import torch

options = onnxruntime.SessionOptions()
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

session = onnxruntime.InferenceSession(r"C:\Users\zhangchao\project\Olive\examples\bert\models\google-bert_bert-base-multilingual-cased\model.onnx",
                                    #    sess_options=options,
                                    #    providers=["CPUExecutionProvider"])
                                       providers=["QNNExecutionProvider"],
                                       provider_options=[{"backend_path": "QnnHtp.dll"}])

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
text = "Hello I'm a [MASK] model."
encoded_input = tokenizer(
                    text,
                    padding="max_length",
                    max_length= 128,
                    truncation=True,
                    return_tensors='pt')

result = session.run(["logits"],
                     {
                        "input_ids": np.array(encoded_input['input_ids']).astype(np.int64),
                        "attention_mask": np.array(encoded_input['attention_mask']).astype(np.int64),
                        "token_type_ids": np.array(encoded_input['token_type_ids']).astype(np.int64)
                    })

outputs = result[0]
masked_index = torch.nonzero(encoded_input['input_ids'].squeeze() == tokenizer.mask_token_id, as_tuple=False).squeeze(-1)

logits = outputs[0, masked_index, :]
probs = torch.from_numpy(logits).softmax(dim=-1)
values, predictions = probs.topk(5)
for v, p in zip(values, predictions):
   print(f"score: {v}, token: {p}, token_str: {tokenizer.decode([p])}")
