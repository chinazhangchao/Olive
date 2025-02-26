from datasets import load_dataset
d=load_dataset("wikipedia", "20220301.en", streaming=True, trust_remote_code=True)
item = next(iter(d['train']))
item['text']
