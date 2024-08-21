import time
import json
from mpi4py import MPI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class LLM_Agent:
    def __init__(self, llm, tokenizer, pipe):
        self.tokenizer = tokenizer
        self.llm = llm
        self.pipe = pipe
        self.yes_ids = self.tokenizer.convert_tokens_to_ids("Yes")
        self.no_ids = self.tokenizer.convert_tokens_to_ids("No")
        # Set pad token and pad token id
        self.tokenizer.pad_token = self.tokenizer.eos_token
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
    def generate_output(self, messages, max_new_tokens=5, **kwargs):
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }
        output = self.pipe(messages, **generation_args)
        generated_texts = output[0]['generated_text']
        return generated_texts

def main():
    model_id = "microsoft/Phi-3.5-MoE-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    assistance = LLM_Agent(model, tokenizer, pipe)

    input_text = "What is your favorite color?"
    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": input_text},
    ]

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Only rank 0 generates the output
    if rank == 0:
        prob = assistance.generate_output(messages, max_new_tokens=5)
        print(f'{prob}')
        with open("output_file.txt", "w") as f:
            f.write(f"Question: {input_text}\nResponse: {prob}\n\n")
    else:
        prob = None

if __name__ == "__main__":
    main()
