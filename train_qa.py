from squadv2_dataset import SquadV2Dataset
from prompt_tuning import GPT2PromptTuneModel
from transformers import TrainingArguments, Trainer
import pickle

model_name = "gpt2"

print("loading train dataset")
try:
    raise Exception
    with open("cached_summarisation_train.bin", "rb") as f:
        train_dataset = pickle.load(f)
except:
    train_dataset = SquadV2Dataset(
        model_name, "squadv2/train-v2.0.json", 750, 1022 - 750 - 100, 100, 10_000)
    with open("cached_squad_train.bin", "wb") as f:
        pickle.dump(train_dataset, f)
print("loaded train dataset")

print("loading eval dataset")
val_dataset = SquadV2Dataset(
    model_name, "squadv2/dev-v2.0.json", 750, 1023 - 750 - 100, 100, 1_000)
print("loaded eval dataset")


print("loading model")

model = GPT2PromptTuneModel.from_pretrained(
    model_name)
model.init_prompt_tuning(soft_prompt_token_ids=train_dataset.tokenizer.encode(
    "[ANSWER]", add_special_tokens=False))
print("loaded model")


args = TrainingArguments(weight_decay=1e-4, output_dir="qa_model", evaluation_strategy="steps", eval_steps=500,
                         per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=2, save_strategy="steps", save_total_limit=2, load_best_model_at_end=True, num_train_epochs=10)

trainer = Trainer(model=model, args=args, train_dataset=train_dataset,
                  eval_dataset=val_dataset, data_collator=train_dataset.collate)

print("training")
trainer.train()
print("trained")

trainer.save_model(output_dir="qa_best")
