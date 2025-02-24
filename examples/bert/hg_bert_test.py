from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-multilingual-cased')
ret = unmasker("Hello I'm a [MASK] model.")
print(ret)
# [{'score': 0.10182079672813416, 'token': 13192, 'token_str': 'model', 'sequence': "Hello I'm a model model."}, {'score': 0.05212637409567833, 'token': 11356, 'token_str': 'world', 'sequence': "Hello I'm a world model."}, {'score': 0.048930246382951736, 'token': 11165, 'token_str': 'data', 'sequence': "Hello I'm a data model."}, {'score': 0.02036013826727867, 'token': 23578, 'token_str': 'flight', 'sequence': "Hello I'm a flight model."}, {'score': 0.020079592242836952, 'token': 14155, 'token_str': 'business', 'sequence': "Hello I'm a business model."}]
