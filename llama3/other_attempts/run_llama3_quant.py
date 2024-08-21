import time, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import re
import accelerate
#from accelerate import init_empty_weights, load_checkpoint_and_dispatch


class LLM_Agent:
    def __init__(self, llm, tokenizer):
        self.tokenizer = tokenizer
        self.llm = llm
        self.yes_ids = self.tokenizer.convert_tokens_to_ids("Yes")
        self.no_ids = self.tokenizer.convert_tokens_to_ids("No")
        
        # Set pad token and pad token id
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.pad_token = self.tokenizer.pad_token
        self.tokenizer.padding_side = "left"

    @torch.no_grad()
    def output_logit(self, input_text, **kwargs):
        inputs = self.tokenizer(input_text, padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.cuda()
        logits = self.llm(input_ids=input_ids).logits[:, -1]
        yes_prob = logits[:, self.yes_ids]
        no_prob = logits[:, self.no_ids]
        logit_value = yes_prob - no_prob
        return logit_value.cpu()

    @torch.no_grad()
    def generate_output(self, input_text, max_new_tokens=400):
        print(f'input text is: {input_text}')
        inputs = self.tokenizer(input_text, padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.llm.device)
        print('about to generate')
        output = self.llm.generate(input_ids, do_sample=False, max_new_tokens=max_new_tokens)

        print('generation complete')
        generated_texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        print(f'output: {generated_texts}')

        return generated_texts


def main():
    model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
    tokenizer_id = model_id
    #model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_8bit = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    #print(model.hf_device_map)

    assistance = LLM_Agent(model_8bit, tokenizer)

    input_text = "Who was Quanah Parker, the comanche?"

    question = input_text
    prob = assistance.generate_output(question, max_new_tokens=400)


if __name__ == "__main__":
    main()

