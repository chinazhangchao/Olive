# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

from logging import getLogger
from datasets import load_dataset
import numpy as np
from torch.utils.data import (DataLoader, Dataset)
from transformers import AutoTokenizer
from olive.data.registry import Registry

logger = getLogger(__name__)

class SquadV2DataReader(Dataset):
    def __init__(self, split='train') -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-large-uncased-whole-word-masking-finetuned-squad")
        self.fixed_length = self.tokenizer.model_max_length
        self.dataset = load_dataset("rajpurkar/squad_v2", split=split)
        self.datasize = 10

    def preprocess(self, question, context, fixed_length=None):
        if fixed_length is not None:
            assert fixed_length > 0
            fixed_length = min(fixed_length, self.tokenizer.model_max_length)
            params = {
                'padding': 'max_length',
                'truncation': True,
                'max_length': fixed_length,
                'return_tensors': 'np'
            }
        else:
            params = {
                'return_tensors': 'np'
            }
            
        encoded_input = self.tokenizer(
            question,
            context,
            **params
        )
        tokens = self.tokenizer.convert_ids_to_tokens(list(encoded_input.input_ids.reshape(-1)))
        return (encoded_input.input_ids, encoded_input.attention_mask, encoded_input.token_type_ids, tokens)

    def __len__(self) -> int:
        return self.datasize

    def __getitem__(self, idx) -> tuple[dict[str, np.array], str]:
      input_ids, attention_mask, token_type_ids, _tokens = self.preprocess(
            self.dataset[idx]["question"],
            self.dataset[idx]["context"],
            fixed_length=self.fixed_length,
        )
      model_inputs = {
          'input_ids':   np.array(input_ids, dtype=np.int64),
          'attention_mask':  np.array(attention_mask, dtype=np.int64),
      }

      return model_inputs, self.dataset[idx]["answers"]["text"]

@Registry.register_dataloader()
def squad_calibration_reader(dataset, batch_size,
                                 **kwargs) -> DataLoader:
  dataset = SquadV2DataReader()
  return DataLoader(dataset, batch_size=batch_size, shuffle=False)
