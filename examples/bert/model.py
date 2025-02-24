
from safetensors import safe_open

tensors = {}
with safe_open(r"C:\Users\zhangchao\.cache\huggingface\hub\models--google-bert--bert-base-multilingual-cased\snapshots\3f076fdb1ab68d5b2880cb87a0886f315b8146f8\model.safetensors", framework="pt", device="cpu") as f:
   # 获取模型参数的名字
    print(list(f.keys()))

    # 获取元数据
    metadata = f.metadata()
    for key, value in metadata.items():
        print(f"{key}: {value}")
