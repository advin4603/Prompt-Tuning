from transformers import GPT2TokenizerFast
from prompt_tuning import GPT2PromptTuneModel
import torch
from summarisation_dataset import SummarisationDataset
from torch.utils.data import DataLoader
from rich.progress import track
import evaluate

model_name = "gpt2"
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
model = GPT2PromptTuneModel.from_pretrained(
    "summarisation_best", True).to("cpu")

test_dataset = SummarisationDataset(
    model_name, "cnn_dailymail/test.csv", 750, 1023 - 750, 500)
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
