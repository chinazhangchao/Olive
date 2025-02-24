from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained("bert-base-multilingual-cased")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
print(output.keys())
# 'last_hidden_state', 'pooler_output'
# 将输出转换为可读字符串
last_hidden_state = output['last_hidden_state']
pooler_output = output['pooler_output']

# 使用 tokenizer 将张量转换为可读字符串
decoded_last_hidden_state = tokenizer.decode(last_hidden_state[0])
decoded_pooler_output = tokenizer.decode(pooler_output[0])

print("Last Hidden State:", decoded_last_hidden_state)
print("Pooler Output:", decoded_pooler_output)
