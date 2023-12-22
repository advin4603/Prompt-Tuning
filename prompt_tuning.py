from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import nn
import torch
from typing import Optional, Tuple
import os


class GPT2PromptTuneModel(GPT2LMHeadModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, has_soft_prompt=False, *args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        for param in model.parameters():
            param.requires_grad = False
        if has_soft_prompt:
            print("IGNORE ABOVE WARNING :p")
            soft_prompt_state = torch.load(os.path.join(
                pretrained_model_name_or_path, "pytorch_model.bin"))["soft_prompt_layer.weight"]
            model.soft_prompt_layer = nn.Embedding.from_pretrained(
                soft_prompt_state)
            model.soft_prompt_count = soft_prompt_state.shape[0]
            model.add_prompt_mode = True

        return model

    def init_prompt_tuning(self, soft_prompt_count=None, soft_prompt_token_ids=None):
        # Freeze all model parameters

        for param in self.parameters():
            param.requires_grad = False

        if soft_prompt_count is None and soft_prompt_token_ids is None:
            raise ValueError(
                "Both soft prompt count and soft prompt token ids cant be None")
        elif soft_prompt_count is not None and soft_prompt_token_ids is not None:
            raise ValueError(
                "Both soft prompt count and soft prompt token ids cannot be provided")
        if soft_prompt_count is None:
            self.soft_prompt_count = len(soft_prompt_token_ids)
            pretrained_embeddings = []
            for i in soft_prompt_token_ids:
                pretrained_embeddings.append(
                    self.transformer.wte.weight[i].tolist())
            self.soft_prompt_layer = nn.Embedding.from_pretrained(
                torch.tensor(pretrained_embeddings), freeze=False)
        else:
            self.soft_prompt_count = soft_prompt_count

            self.soft_prompt_layer = nn.Embedding(
                soft_prompt_count, self.transformer.embed_dim)
        self.add_prompt_mode = True
        self.soft_prompt_token_ids = torch.tensor(soft_prompt_token_ids) if soft_prompt_token_ids is not None else torch.zeros(
            soft_prompt_count, dtype=torch.long)

        for block in self.transformer.h:
            block.attn.bias = torch.tril(torch.ones((1024+self.soft_prompt_count, 1024+self.soft_prompt_count), dtype=torch.bool)).view(
                1, 1, 1024+self.soft_prompt_count, 1024+self.soft_prompt_count
            )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        if not self.add_prompt_mode:
            return super().forward(
                input_ids,
                past_key_values,
                attention_mask,
                token_type_ids,
                position_ids,
                head_mask,
                inputs_embeds,
                encoder_hidden_states,
                encoder_attention_mask,
                labels,
                use_cache,
                output_attentions,
                output_hidden_states,
                return_dict,

            )
        soft_prompt_embeddings = self.soft_prompt_layer(
            torch.arange(self.soft_prompt_count).to(self.device)).unsqueeze(0).expand(input_ids.shape[0], -1, -1)

        if input_ids is None:
            text_embeddings = inputs_embeds
        else:
            text_embeddings = self.transformer.wte(input_ids)
        if attention_mask is not None:
            attention_mask = torch.cat(
                (torch.ones(attention_mask.shape[0], self.soft_prompt_count).to(self.device), attention_mask), dim=-1)

        if token_type_ids is not None:
            token_type_ids = torch.cat(
                (torch.zeros(
                    token_type_ids.shape[0], self.soft_prompt_count, dtype=torch.long).to(self.device), token_type_ids), dim=-1
            )

        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1]).to(self.device).unsqueeze(
                0).expand(input_ids.shape[0], -1)

        position_ids = torch.cat((
            torch.zeros(
                position_ids.shape[0], self.soft_prompt_count, dtype=torch.long).to(self.device), position_ids), dim=-1)
        t = position_ids.to("cpu")
        if labels is not None:
            labels = torch.cat((
                -100 * torch.ones(
                    labels.shape[0], self.soft_prompt_count, dtype=torch.long).to(self.device), labels), dim=-1)

        inputs_embeds = torch.cat(
            (soft_prompt_embeddings, text_embeddings), dim=1)

        return super().forward(
            None,
            past_key_values,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )

    def generate(self, input_ids, display_prompt=False, **kwargs):

        self.add_prompt_mode = False
        soft_prompt_embeddings = self.soft_prompt_layer(
            torch.arange(self.soft_prompt_count).to(self.device)).unsqueeze(0).expand(input_ids.shape[0], -1, -1)

        if input_ids is None:
            text_embeddings = kwargs.pop("inputs_embeds")
        else:
            text_embeddings = self.transformer.wte(input_ids)

        attention_mask = kwargs.pop("attention_mask", None)
        position_ids = kwargs.pop("position_ids", None)

        if attention_mask is not None:
            attention_mask = torch.cat(
                (torch.ones(attention_mask.shape[0], self.soft_prompt_count).to(self.device), attention_mask), dim=-1)

        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1]).to(self.device).unsqueeze(
                0).expand(input_ids.shape[0], -1)

        position_ids = torch.cat((
            torch.zeros(
                position_ids.shape[0], self.soft_prompt_count, dtype=torch.long).to(self.device), position_ids), dim=-1)

        inputs_embeds = torch.cat(
            (soft_prompt_embeddings, text_embeddings), dim=1)

        # if input_ids is not None:
        #     input_ids = torch.cat((self.soft_prompt_token_ids.to(self.device).unsqueeze(
        #         0).expand(input_ids.shape[0], -1), input_ids), dim=-1)

        generate_output = super().generate(
            input_ids if display_prompt else None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        self.add_prompt_mode = True
        return generate_output


if __name__ == "__main__":
    # Load the GPT-2 tokenizer and model
    model_name = "gpt2"  # You can use "gpt2" or "gpt2-small" for the smallest version
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    prompt_text = "Once upon a time"
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    model = GPT2PromptTuneModel.from_pretrained(
        model_name)
    model.init_prompt_tuning(soft_prompt_token_ids=prompt_ids)

    # Input text for inference
    input_text = " "

    # Encode the input text
    input_ids = tokenizer.encode(
        input_text, return_tensors="pt", add_special_tokens=False)

    # Generate text based on the input
    output = model.generate(input_ids, max_length=100, num_return_sequences=1,
                            no_repeat_ngram_size=2, top_k=50, top_p=0.95, do_sample=True, display_prompt=True)

    # Decode and print the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated Text:\n", generated_text)
