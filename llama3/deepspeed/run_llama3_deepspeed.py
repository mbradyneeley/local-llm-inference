from transformers import LlamaForCausalLM, AutoTokenizer, AutoModel

import deepspeed
import torch

# Specify the model name
model_name = "meta-llama/Meta-Llama-3-70B"

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the model
model = LlamaForCausalLM.from_pretrained(model_name)



# Initialize the DeepSpeed-Inference engine
ds_engine = deepspeed.init_inference(model,
                                 mp_size=8,
                                 dtype=torch.bfloat16)
model = ds_engine.module
output = model('Tell me a pirate joke')
