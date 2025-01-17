from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('Intel/bert-base-uncased-mrpc')
model = BertModel.from_pretrained("Intel/bert-base-uncased-mrpc")
text = "The inspector analyzed the soundness in the building."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
# print BaseModelOutputWithPoolingAndCrossAttentions and  pooler_output

# Print tokens * ids in of inmput string below
print('Tokenized Text: ', tokenizer.tokenize(text), '\n')
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)))

#Print tokens in text
print(encoded_input['input_ids'][0])
result = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'][0])
print(result)
