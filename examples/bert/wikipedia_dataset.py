from datasets import load_dataset
iterable_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
iterable_dataset=iterable_dataset.skip(2)
it = iter(iterable_dataset)
for i in range(0, 3):
    item = next(it)
    print(item)
