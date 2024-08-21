import logging
import os
import time
import fire
import pandas as pd
import torch
from accelerate import Accelerator
from peft import PeftModel
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def get_configs():
    generation_cfg = {
        'max_new_tokens': 1024,
        'temperature': 0.8,
        'top_p': 0.95,
        'repetition_penalty': 1.15,
        'do_sample': True
    }
    model_cfg = {
        'pad_to_multiple_of': 8,
    }
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    prompt_config = {
        'system_format': "### System:\n{system}\n\n",
        'system_no_input_prompt': "Below is an instruction that describes a task. Write a response "
                                  "that appropriately completes the request.\n\n",
        'turn_no_input_format': "### Instruction:\n{instruction}\n\n### Response:\n"
    }

    return generation_cfg, model_cfg, bnb_config, prompt_config

def get_input_prompt(instruction, prompt_config):
    res = prompt_config['system_format'].format(system=prompt_config['system_no_input_prompt']) \
        + prompt_config['turn_no_input_format'].format(instruction=instruction)
    return res

def run_generation(generation_cfg, input_prompt, tokenizer, model, accelerator):
    model = accelerator.prepare(model)
    accelerator.wait_for_everyone()

    start_time = time.time()

    with torch.inference_mode():
        input_ids = tokenizer(input_prompt, return_tensors='pt').input_ids.to(accelerator.device)
        unwrapped_model = accelerator.unwrap_model(model)

        generated_tokens = unwrapped_model.generate(input_ids, **generation_cfg)
        generated_tokens = generated_tokens.cpu().tolist()

    outputs = [tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_tokens]

    generation_end_time = time.time()
    logger.info(f"Generation time: {generation_end_time - start_time} sec")
    return outputs

def get_model(model_name, model_cfg, bnb_config, accelerator):
    with accelerator.main_process_first():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True  # Optimize memory usage during model loading
        )
        model.resize_token_embeddings(model.config.vocab_size + 1,
                                      pad_to_multiple_of=model_cfg['pad_to_multiple_of'])
        model.gradient_checkpointing_enable()  # Enable gradient checkpointing
        return model

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              padding_side='left',
                                              truncation=True)

    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )

    return tokenizer

def main(model_name,
         instruction,
         output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    generation_cfg, model_cfg, bnb_config, prompt_config = get_configs()

    accelerator = Accelerator()

    logger.info("Loading tokenizer")
    tokenizer = get_tokenizer(model_name)

    logger.info("Loading model")
    model = get_model(model_name, model_cfg, bnb_config, accelerator)

    logger.info("Starting generation")
    input_prompt = get_input_prompt(instruction, prompt_config)
    output_sequences = run_generation(generation_cfg, input_prompt, tokenizer, model, accelerator)

    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        output_df = pd.DataFrame(output_sequences, columns=["response"])
        output_file_path = os.path.join(output_dir, "result.jsonl")
        output_df.to_json(output_file_path, orient="records", lines=True)

if __name__ == '__main__':
    fire.Fire(main)

