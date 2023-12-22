from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast
import csv
from rich.progress import track
import torch
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100


class EuroparlDataset(Dataset):
    def __init__(self, model_name: str, english_sequences: list[str], german_sequences: list[str], english_max_length: int, german_max_length: int):
        super().__init__()
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.english_sequences = english_sequences
        self.german_sequences = german_sequences

        tokenizer_batch_size = 128
        self.english_sequences_ids = []
        for i in track(range(0, len(self.english_sequences), tokenizer_batch_size), description="tokenizing english"):
            batch = self.english_sequences[i:i+tokenizer_batch_size]
            self.english_sequences_ids.extend(self.tokenizer(batch, max_length=english_max_length, return_attention_mask=False,
                                                             return_token_type_ids=False, truncation=True).input_ids)

        self.german_sequences_ids = []
        for i in track(range(0, len(self.german_sequences), tokenizer_batch_size), description="tokenizing german"):
            batch = self.german_sequences[i:i+tokenizer_batch_size]
            self.german_sequences_ids.extend(self.tokenizer(batch, max_length=german_max_length, return_attention_mask=False,
                                                            return_token_type_ids=False, truncation=True).input_ids)

    def __len__(self):
        return len(self.english_sequences)

    def __getitem__(self, index):
        english_ids = self.english_sequences_ids[index]
        german_ids = self.german_sequences_ids[index]
        separator_id = self.tokenizer.bos_token_id
        sequence_ids = torch.tensor(
            [*english_ids, separator_id, *german_ids]
        )

        labels = torch.tensor(
            [*([IGNORE_INDEX] * len(english_ids)), IGNORE_INDEX, *german_ids]
        )

        return {"input_ids": sequence_ids, "labels": labels}

    def collate(self, item_list):
        input_ids = pad_sequence([i["input_ids"] for i in item_list],
                                 batch_first=True, padding_value=self.tokenizer.eos_token_id)
        labels = pad_sequence([i["labels"] for i in item_list],
                              batch_first=True, padding_value=self.tokenizer.eos_token_id)
        attention_mask = pad_sequence(
            [torch.ones(i["input_ids"].shape) for i in item_list], batch_first=True, padding_value=0)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
