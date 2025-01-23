
from typing import Dict, Union

import torch
from datasets import load_dataset
from evaluate import load
from transformers import AutoTokenizer

from olive.data.component.dataset import BaseDataset
from olive.data.registry import Registry


@Registry.register_pre_process()
def bert_squad_pre_process(dataset, model_name, input_cols, label_col="label",
                           seq_length=512, max_samples=None, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def generate_inputs(samples, indices):
        encoded_input = tokenizer(
            *[samples[input_col] for input_col in input_cols],
            padding="max_length",
            max_length=seq_length,
            truncation=True,
            add_special_tokens=True
        )
        encoded_input[label_col] = indices
        return encoded_input

    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    tokenized_datasets = dataset.map(
        generate_inputs,
        batched=True,
        with_indices=True,
        remove_columns=dataset.column_names,
    )
    tokenized_datasets.set_format("torch", output_all_columns=True)

    return BaseDataset(tokenized_datasets, label_col, max_samples)


@Registry.register_post_process()
def bert_squad_post_process(outputs, **kwargs):
    logits = [outputs["start_logits"], outputs["end_logits"]]
    return torch.stack(logits, dim=1)


def eval_squad(
    outputs,
    targets,
    dataset_config: Dict[str, str],
    model_name: str,
    seq_length: int = 512,
) -> Dict[str, Union[float, int]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = load_dataset(
        path=dataset_config["data_name"],
        split=dataset_config["split"],
    )

    predictions = []
    references = []

    for pred, i in zip(outputs.preds, targets):
        sample = dataset[i.item()]
        encoded_input = tokenizer(
            sample["question"],
            sample["context"],
            padding="max_length",
            max_length=seq_length,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt"
        )

        start_logits, end_logits = pred.unbind(dim=0)
        start_index = start_logits.argmax(dim=-1)
        end_index = end_logits.argmax(dim=-1)
        offset_mapping = encoded_input["offset_mapping"]
        answer_start = offset_mapping[:, start_index, 0].squeeze()
        answer_end = offset_mapping[:, end_index, 1].squeeze()
        pred_answer = sample["context"][answer_start:answer_end]

        references.append({
            "id": sample["id"],
            "answers": {
                "answer_start": sample["answers"]["answer_start"],
                "text": sample["answers"]["text"],
            },
        })
        predictions.append({
            "id": sample["id"],
            "prediction_text": pred_answer,
        })

    results = load("squad").compute(
        predictions=predictions,
        references=references,
    )

    return {"f1": results["f1"], "exact_match": results["exact_match"]}
