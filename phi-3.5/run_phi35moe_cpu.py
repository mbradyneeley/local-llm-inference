import time
import json
from mpi4py import MPI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

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
        input_ids = inputs.input_ids  # Keep it on CPU
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

        # Start timing
        start_time = time.time()

        # Generate tokens with tqdm progress bar
        output = None
        with tqdm(total=max_new_tokens, desc="Generating tokens", unit="token") as pbar:
            output = self.pipe(messages, **generation_args)
            pbar.update(max_new_tokens)

        # Stop timing
        end_time = time.time()

        # Calculate the number of tokens generated
        generated_texts = output[0]['generated_text']
        token_count = len(self.tokenizer.tokenize(generated_texts))

        # Calculate tokens per second
        tokens_per_second = token_count / (end_time - start_time)
        print(f'Tokens per second: {tokens_per_second:.2f}')

        return generated_texts

def main():
    model_id = "microsoft/Phi-3.5-MoE-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cpu",  # Keep model on CPU
        torch_dtype="auto",
        trust_remote_code=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
        # Removed `device` argument
    )

    assistance = LLM_Agent(model, tokenizer, pipe)

    input_text = "A patient showed these phenotypes: Microcephaly,Delayed speech and language development,Abnormality of toe,Prominent nasal bridge,2-3 toe syndactyly,Overlapping toe,Abnormality of skin pigmentation,Pectus excavatum. The patient's causative variant was in the gene: RTTN. What are the assocations?"
    messages = [
        {"role": "system", "content": "System: You are an expert in clinical human genetics and molecular biology. Your key responsibility is to identify the most significant phenotypes relevant to a patient's diagnosis. Scrutinize any phenotypes that may be unexpectedly associated with a known condition, and make a clear note of these when they arise. Carefully assess traits that appear overly general or common, determining whether they are likely coincidental or truly connected to the condition. Your analysis must be exceptionally precise, as your insights are vital for enhancing disease understanding and shaping accurate diagnoses."},
        {"role": "user", "content": input_text},
    ]

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Only rank 0 generates the output
    if rank == 0:
        prob = assistance.generate_output(messages, max_new_tokens=500)
        print(f'{prob}')
        with open("output_file.txt", "w") as f:
            f.write(f"Question: {input_text}\nResponse: {prob}\n\n")
    else:
        prob = None

if __name__ == "__main__":
    main()
