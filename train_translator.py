from europarl_dataset import EuroparlDataset
from prompt_tuning import GPT2PromptTuneModel
from transformers import TrainingArguments, Trainer
import pickle
import random
import torch

model_name = "gpt2"

with open("europarl-v7.de-en.en") as english_file, open("europarl-v7.de-en.de") as deutch_file:
    pairs = []
    for english_line, deutch_line in zip(english_file, deutch_file):
        pairs.append((english_line.strip(), deutch_line.strip()))

random.seed(42)
random.shuffle(pairs)
pairs = pairs[:5000]
english_lines = [i[0] for i in pairs]
deutch_lines = [i[1] for i in pairs]

train_split = .75
train_length = int(train_split * len(english_lines))
train_english_lines = english_lines[:train_length]
train_deutch_lines = deutch_lines[:train_length]

val_english_lines = english_lines[train_length:]
val_deutch_lines = deutch_lines[train_length:]

print("loading train dataset")

train_dataset = EuroparlDataset(
    model_name, train_english_lines, train_deutch_lines, 511, 512)
print("loaded train dataset")

print("loading eval dataset")
val_dataset = EuroparlDataset(
    model_name, val_english_lines, val_deutch_lines, 511, 512)
print("loaded eval dataset")


print("loading model")

model = GPT2PromptTuneModel.from_pretrained(
    model_name)
model.init_prompt_tuning(soft_prompt_token_ids=train_dataset.tokenizer.encode(
    "[TRANSLATE_EN_DE]", add_special_tokens=False))
print("loaded model")


args = TrainingArguments(output_dir="translation_model", evaluation_strategy="steps", eval_steps=500,
                         per_device_train_batch_size=1, per_device_eval_batch_size=1, save_strategy="steps", save_total_limit=2, load_best_model_at_end=True, num_train_epochs=10)

trainer = Trainer(model=model, args=args, train_dataset=train_dataset,
                  eval_dataset=val_dataset, data_collator=train_dataset.collate)


print("training")
trainer.train()
print("trained")

trainer.save_model(output_dir="translation_best")
