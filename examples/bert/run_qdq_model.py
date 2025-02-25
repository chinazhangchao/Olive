# run_qdq_model.py

import onnxruntime
import numpy as np
from transformers import BertTokenizer

options = onnxruntime.SessionOptions()
options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")

session = onnxruntime.InferenceSession(r"C:\Users\zhangchao\project\Olive\examples\bert\models\google-bert_bert-base-multilingual-cased\model.onnx",
                                    #    sess_options=options,
                                       providers=["CPUExecutionProvider"])
                                    #    providers=["QNNExecutionProvider"],
                                    #    provider_options=[{"backend_path": "QnnHtp.dll"}])

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
                        "input_ids": encoded_input['input_ids'],
                        "attention_mask": encoded_input['attention_mask'],
                        "token_type_ids": encoded_input['token_type_ids']
                    })

# Print output.
print(result)
