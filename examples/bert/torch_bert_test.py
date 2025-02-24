from transformers import BertTokenizer, BertForMaskedLM
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")
# text = "Replace me by any text you'd like."
text = "Hello I'm a [MASK] model."
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)
output = model(**encoded_input)
print(output)
print(output.keys())

masked_index = torch.nonzero(encoded_input['input_ids'] == tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
# Fill mask pipeline supports only one ${mask_token} per sample

logits = output[0, masked_index, :]
probs = logits.softmax(dim=-1)

values, predictions = probs.topk(5)
