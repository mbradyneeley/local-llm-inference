import time, json
from mpi4py import MPI
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
#from accelerate import init_empty_weights, load_checkpoint_and_dispatch


class LLM_Agent:
    def __init__(self, llm, tokenizer):
        self.tokenizer = tokenizer
        self.llm = llm
        self.yes_ids = self.tokenizer.convert_tokens_to_ids("Yes")
        self.no_ids = self.tokenizer.convert_tokens_to_ids("No")
        ## Set pad token and pad token id
        #if self.tokenizer.pad_token is None:
        #    #self.tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # How Guanchu had it set
        #self.tokenizer.add_special_tokens({"pad_token":"<pad>"}) # Comes from https://huggingface.co/docs/transformers/main/en/model_doc/llama3
        self.tokenizer.pad_token = self.tokenizer.eos_token
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
    def generate_output(self, input_text, max_new_tokens=5):
        print(f'input text is: {input_text}')
        inputs = self.tokenizer(input_text, padding=True, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.llm.device)
        print('about to generate')
        output = self.llm.generate(input_ids, do_sample=True, max_new_tokens=max_new_tokens)

        print('generation complete')
        generated_texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        #print(f'output: {generated_texts}')
        print("done")

        return generated_texts


def main():
    model_id = "meta-llama/Meta-Llama-3-8B"
    tokenizer_id = model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    # TODO: Coulnt fix so went with EOS. Error wouldnt let me resize vocab. Add the padding token and resize the model embeddings
    #tokenizer.add_special_tokens({"pad_token": "<pad>"})
    tokenizer.pad_token = tokenizer.eos_token


    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    # To account for the padding token we added https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/inference/local_inference
    # Debug: Check the model's vocab size before resizing
    print(f'Model vocab size before resizing: {model.config.vocab_size}')

    # Resize model embeddings based on the updated tokenizer size
    model.config.pad_token_id = tokenizer.pad_token_id
    # No resize necessary now when using eos token
    #model.resize_token_embeddings(len(tokenizer))

    # Debug: Check the model's vocab size after resizing
    print(f'Model vocab size after resizing: {model.config.vocab_size}')
    #print(model.hf_device_map)

    assistance = LLM_Agent(model, tokenizer)

    input_text = "What is your favorite color?"

    # Format according to Llama3 prompt formatting (NOT instruct)
    question = "<|begin_of_text|>{{{{ {} }}}}".format(input_text)
    #prob = assistance.generate_output(question, max_new_tokens=400)

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Only rank 0 generates the output
    if rank == 0:
        prob = assistance.generate_output(question, max_new_tokens=5)
        print(f'{prob}')
        with open("output_file.txt", "w") as f:
            f.write(f"Question: {question}\nResponse: {prob}\n\n")
    else:
        prob = None


    #if rank == 0:
    #    print(f'{prob}')
    #    with open("output_file.txt", "w") as f:
    #        f.write(f"Question: {question}\nResponse: {prob}\n\n")

if __name__ == "__main__":
    main()

