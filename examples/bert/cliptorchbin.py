from transformers import AutoModel

model = AutoModel.from_pretrained(r"C:\Users\zhangchao\.cache\huggingface\hub\models--openai--clip-vit-base-patch32\snapshots\3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
print(model)
