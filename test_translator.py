from transformers import GPT2TokenizerFast
from prompt_tuning import GPT2PromptTuneModel
import torch
from europarl_dataset import EuroparlDataset
from torch.utils.data import DataLoader
from rich.progress import track
import evaluate
import random

model_name = "gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2PromptTuneModel.from_pretrained(
    "translation_best", True).to("cpu")


with open("europarl-v7.de-en.en") as english_file, open("europarl-v7.de-en.de") as deutch_file:
    pairs = []
    for english_line, deutch_line in zip(english_file, deutch_file):
        pairs.append((english_line.strip(), deutch_line.strip()))

random.seed(42)
random.shuffle(pairs)
pairs = pairs[-500:]
english_lines = [i[0] for i in pairs]
deutch_lines = [i[1] for i in pairs]


test_dataset = EuroparlDataset(
    model_name, english_lines, deutch_lines, 511, 512)
test_dataloader = DataLoader(
    test_dataset, batch_size=16, collate_fn=test_dataset.collate)
model.eval()

rouge = evaluate.load("rouge")
rouge_l_sum = 0
count = 0
with torch.no_grad():
    for data in track(test_dataloader, description="computing metrics"):
        labels = data.pop("labels")
        # for k in data:
        #     data[k] = data[k].to("cuda:0")
        outputs = model.generate(**data, top_k=50, top_p=0.95,
                                 temperature=0.1, no_repeat_ngram_size=4, do_sample=True, max_new_tokens=50)
        for i, output in enumerate(outputs):
            target = labels[i]
            output = output[(output != test_dataset.tokenizer.eos_token_id)]
            target = target[(target != -100) & (target !=
                                                test_dataset.tokenizer.eos_token_id)]
            output_text = tokenizer.decode(output, skip_special_tokens=True)
            target_text = tokenizer.decode(target, skip_special_tokens=True)
            print(output_text)
            print("-"*50)
            print(target_text)
            results = rouge.compute(
                predictions=[output_text], references=[[target_text]])
            print(results)
            print("="*50)
            rouge_l_sum += results["rougeL"]
            count += 1

print("average RougeL:", rouge_l_sum/count)
