from torch.utils.data import Dataset
from transformers import GPT2TokenizerFast
import json
from rich.progress import track
import torch
from torch.nn.utils.rnn import pad_sequence
import random

IGNORE_INDEX = -100


class SquadV2Dataset(Dataset):
    def __init__(self, model_name: str, data_path: str, context_max_length: int, question_max_length: int, answer_max_length: int, max_data_size: int | None = None):
        super().__init__()
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        with open(data_path) as f:
            data = json.load(f)
        self.contexts = []
        self.questions = []
        self.answers = []
        self.context_indices = []
        for topic in data["data"]:
            for paragraph in topic["paragraphs"]:
                self.contexts.append(paragraph["context"])
                context_index = len(self.contexts) - 1
                for qa in paragraph["qas"]:
                    if len(qa["answers"]) > 0:
                        self.questions.append(qa["question"])
                        # TODO maybe not take only one answer

                        self.answers.append(qa["answers"][0]["text"])
                        self.context_indices.append(context_index)
        tokenizer_batch_size = 128

        self.contexts_ids = []
        for i in track(range(0, len(self.contexts), tokenizer_batch_size), description="tokenizing contexts"):
            batch = self.contexts[i: i + tokenizer_batch_size]
            self.contexts_ids.extend(self.tokenizer(batch, max_length=context_max_length,
                                     return_attention_mask=False, return_token_type_ids=False, truncation=True).input_ids)

        self.question_ids = []
        self.answer_ids = []
        for i in track(range(0, len(self.questions), tokenizer_batch_size), description="tokenizing questions and answers"):
            question_batch = self.questions[i: i + tokenizer_batch_size]
            self.question_ids.extend(self.tokenizer(question_batch, max_length=question_max_length,
                                     return_attention_mask=False, return_token_type_ids=False, truncation=True).input_ids)

            answer_batch = self.answers[i: i + tokenizer_batch_size]
            self.answer_ids.extend(self.tokenizer(answer_batch, max_length=answer_max_length,
                                                  return_attention_mask=False, return_token_type_ids=False, truncation=True).input_ids)

        zipped_items = list(
            zip(self.context_indices, self.question_ids, self.answer_ids))
        random.shuffle(zipped_items)
        if max_data_size is not None:
            zipped_items = zipped_items[:max_data_size]
        self.context_indices = [i[0] for i in zipped_items]
        self.question_ids = [i[1] for i in zipped_items]
        self.answer_ids = [i[2] for i in zipped_items]

    def __len__(self):
        return len(self.question_ids)

    def __getitem__(self, index):
        context_ids = self.contexts_ids[self.context_indices[index]]
        question_ids = self.question_ids[index]
        answer_ids = self.answer_ids[index]
        separator_id = self.tokenizer.bos_token_id
        sequence_ids = torch.tensor(
            [*context_ids, separator_id, *question_ids, separator_id, *answer_ids])
        labels = torch.tensor(
            [
                *([IGNORE_INDEX] * len(context_ids)),
                IGNORE_INDEX,
                *([IGNORE_INDEX] * len(question_ids)),
                IGNORE_INDEX,
                *answer_ids
            ]
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
