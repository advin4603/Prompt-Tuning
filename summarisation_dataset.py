from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast
import csv
from rich.progress import track
import torch
from torch.nn.utils.rnn import pad_sequence
import random

IGNORE_INDEX = -100


class SummarisationDataset(Dataset):
    def __init__(self, model_name: str, data_path: str, article_max_length: int, highlight_max_length: int, max_data_length: int | None = None):
        super().__init__()
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.articles, self.highlights = [], []
        self.max_data_length = max_data_length
        with open(data_path) as f:
            csv_reader = csv.DictReader(f)
            rows = [row for row in csv_reader]
        random.shuffle(rows)
        rows = rows[:max_data_length] if max_data_length is not None else rows
        for row in rows:
            self.articles.append(row["article"])
            self.highlights.append(row["highlights"])
        tokenizer_batch_size = 128
        self.articles_input_ids = []
        for i in track(range(0, len(self.articles) - tokenizer_batch_size + 1, tokenizer_batch_size), description="tokenizing articles"):
            batch = self.articles[i:i+tokenizer_batch_size]
            self.articles_input_ids.extend(self.tokenizer(batch, max_length=article_max_length, return_attention_mask=False,
                                                          return_token_type_ids=False, truncation=True).input_ids)

        self.highlights_input_ids = []
        for i in track(range(0, len(self.highlights) - tokenizer_batch_size + 1, tokenizer_batch_size), description="tokenizing highlights"):
            batch = self.highlights[i: i+tokenizer_batch_size]
            self.highlights_input_ids.extend(self.tokenizer(batch, max_length=highlight_max_length, return_attention_mask=False,
                                                            return_token_type_ids=False, truncation=True).input_ids)

    def __len__(self):
        return len(self.articles_input_ids)

    def __getitem__(self, index):
        article_ids = self.articles_input_ids[index]
        highlight_ids = self.highlights_input_ids[index]
        separator_id = self.tokenizer.bos_token_id
        sequence_ids = torch.tensor(
            [*article_ids, separator_id, *highlight_ids])
        labels = torch.tensor(
            [*[IGNORE_INDEX for _ in article_ids], IGNORE_INDEX, *highlight_ids])
        if len(sequence_ids) > 1024:
            raise ValueError
        return {"input_ids": sequence_ids, "labels": labels}

    def collate(self, item_list):
        input_ids = pad_sequence([i["input_ids"] for i in item_list],
                                 batch_first=True, padding_value=self.tokenizer.eos_token_id)
        labels = pad_sequence([i["labels"] for i in item_list],
                              batch_first=True, padding_value=self.tokenizer.eos_token_id)
        attention_mask = pad_sequence(
            [torch.ones(i["input_ids"].shape) for i in item_list], batch_first=True, padding_value=0)

        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
