# TODO: Ran out of memory
import time, json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
from mpi4py import MPI
from accelerate import Accelerator

class LLM_Agent:
    def __init__(self, llm, tokenizer):
        self.tokenizer = tokenizer
        self.llm = llm
        self.yes_ids = self.tokenizer.convert_tokens_to_ids("Yes")
        self.no_ids = self.tokenizer.convert_tokens_to_ids("No")
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
    def generate_output(self, model, input_text, accelerator, max_new_tokens=400):
        model = accelerator.prepare(model)
        print(f'input text is: {input_text}')

        with torch.inference_mode():
            inputs = self.tokenizer(input_text, padding=True, return_tensors="pt")
            input_ids = inputs.input_ids.to(self.llm.device)
            print('about to generate')
            output = self.llm.generate(input_ids, do_sample=True, max_new_tokens=max_new_tokens)
            print('generation complete')
            generated_texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            # the batch sizes will be the same across all workers if working with batch of data
            generated_tokens = accelerator.pad_across_processes(
                generated_texts, dim=1, pad_index=self.tokenizer.pad_token_id)
            # gather the outputs from the workers into the main process
            generated_tokens = accelerator.gather_for_metrics(generated_tokens).cpu().tolist()
            print("done")
            
        return generated_texts

def main():
    accelerator = Accelerator()

    with accelerator.main_process_first():
        model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
        tokenizer_id = model_id
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        tokenizer.pad_token = tokenizer.eos_token

        # Load model without device_map TODO: Do we need this?
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        #model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    
    # Prepare model with accelerator
    # The accelerator.prepare() method prepares data and models for distributed processing by moving them to the GPU’s where they are needed
    model = accelerator.prepare(model)

    assistance = LLM_Agent(model, tokenizer)
    input_text = "Who was Quanah Parker, the comanche?"
    question = "{{{{ {} }}}}".format(input_text)
    prob = assistance.generate_output(model, question, accelerator, max_new_tokens=100)
    
    # The accelerator.wait_for_everyone() call is done in order to ensure that all workers finished their part of the inference
    accelerator.wait_for_everyone()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        accelerator.print(f'{prob}')
        with open("output_file.txt", "w") as f:
            f.write(f"Question: {question}\nResponse: {prob}\n\n")

if __name__ == "__main__":
    main()

