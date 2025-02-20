#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os 
import hashlib

import logging
import random
import sys

import torch
import transformers
import numpy as np 
from transformers import AutoModelForCausalLM, set_seed, AutoTokenizer
import torch.distributed as dist

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
)
from alignment.data import maybe_insert_system_message, is_openai_format

from peft import PeftConfig, PeftModel
from dpo_diagnostician import DPODiagnostician
from typing import Optional, Literal, Dict, List, Union, Tuple
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)

@dataclass
class CurrArgs:
    curr_split: str = field(
        default="first",
        metadata={"help": "Which half of the dataset to use for training ('first' or 'second')"}
    )

    num_eval: int = field(
        default=5,
        metadata={"help": "Number of times to evaluate during training"}
    )

def generate_prompt_id(prompt_text):
    # Convert the prompt text to bytes
    prompt_bytes = prompt_text.encode('utf-8')
    
    # Generate a SHA-256 hash of the prompt text
    sha256_hash = hashlib.sha256(prompt_bytes).hexdigest()
    
    return sha256_hash

def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "dpo"],
    auto_insert_empty_system_msg: bool = True,
):
    if task in ["sft", "generation"]:
        messages = example["messages"]
        # We add an empty system message if there is none
        if auto_insert_empty_system_msg:
            maybe_insert_system_message(messages, tokenizer)
        example["text"] = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True if task == "generation" else False,
        )
    elif task == "rm":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            chosen_messages = example["chosen"]
            rejected_messages = example["rejected"]
            # We add an empty system message if there is none
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(chosen_messages, tokenizer)
                maybe_insert_system_message(rejected_messages, tokenizer)

            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
        else:
            raise ValueError(
                f"Could not format example as dialogue for `rm` task! Require `[chosen, rejected]` keys but found {list(example.keys())}"
            )
    elif task == "dpo":
        if all(k in example.keys() for k in ("chosen", "rejected")):
            if not is_openai_format(example["chosen"]) or not is_openai_format(example["rejected"]):
                raise ValueError(
                    f"Could not format example as dialogue for `{task}` task! Require OpenAI format for all messages"
                )

            # For DPO/ORPO, the inputs are triples of (prompt, chosen, rejected), where `chosen` and `rejected` are the final turn of a dialogue
            # We therefore need to extract the N-1 turns to form the prompt
            if "prompt" in example and is_openai_format(example["prompt"]):
                prompt_messages = example["prompt"]
                chosen_messages = example["chosen"]
                rejected_messages = example["rejected"]
            else:
                prompt_messages = example["chosen"][:-1]
                # Now we extract the final turn to define chosen/rejected responses
                chosen_messages = example["chosen"][-1:]
                rejected_messages = example["rejected"][-1:]

            # Prepend a system message if the first message is not a system message
            if auto_insert_empty_system_msg:
                maybe_insert_system_message(prompt_messages, tokenizer)

            example["text_prompt"] = tokenizer.apply_chat_template(prompt_messages, tokenize=False)
            example["text_chosen"] = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
            if tokenizer.bos_token and example["text_chosen"].startswith(tokenizer.bos_token):
                example["text_chosen"] = example["text_chosen"][len(tokenizer.bos_token):]
            example["text_rejected"] = tokenizer.apply_chat_template(rejected_messages, tokenize=False)
            if tokenizer.bos_token and example["text_rejected"].startswith(tokenizer.bos_token):
                example["text_rejected"] = example["text_rejected"][len(tokenizer.bos_token):]
        else:
            raise ValueError(
                f"Could not format example as dialogue for `{task}` task! Require either the "
                f"`[chosen, rejected]` or `[prompt, chosen, rejected]` keys but found {list(example.keys())}"
            )
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example

def get_cache_path(data_args, training_args, tokenizer, split: str) -> str:
    # Create a unique cache key based on relevant parameters
    cache_params = {
        'dataset_mixer': data_args.dataset_mixer,
        'dataset_config_name': data_args.dataset_configs,
        'split': split,
        'seed': training_args.seed,
        'tokenizer': tokenizer.name_or_path,
        'max_length': training_args.max_length,
        'max_prompt_length': training_args.max_prompt_length,
        'chat_template': tokenizer.chat_template,
    }
    
    # Create a hash of the parameters
    param_str = str(sorted(cache_params.items()))
    cache_hash = hashlib.md5(param_str.encode()).hexdigest()
    
    # Create cache directory if it doesn't exist
    cache_dir = '.cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    return os.path.join(cache_dir, f'{split}_{cache_hash}.cache')

def process_and_cache_dataset(raw_datasets, data_args, tokenizer, split: str, training_args):
    cache_path = get_cache_path(data_args, training_args, tokenizer, split)
    
    # If cache exists, load it
    if os.path.exists(cache_path):
        logger.info(f"Loading cached dataset from {cache_path}")
        return torch.load(cache_path)
    
    logger.info(f"Processing dataset for {split} split")
    dataset = raw_datasets[split]
    
    # Apply existing processing steps
    if 'prompt_id' not in dataset.features:
        dataset = dataset.map(
            lambda x: {"prompt_id": generate_prompt_id(x['chosen'][0]['content'])},
            num_proc=data_args.preprocessing_num_workers,
            desc=f"Creating prompt_id for {split}",
        )
    
    column_names = list(dataset.features)
    if "prompt_id" in column_names:
        column_names.remove("prompt_id")
    
    # Apply chat template
    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "dpo",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )
    
    # Rename columns
    dataset = dataset.rename_columns({
        "text_prompt": "prompt",
        "text_chosen": "chosen",
        "text_rejected": "rejected"
    })
    
    # Filter samples
    dataset = dataset.filter(
        lambda x: x["chosen"] != x["rejected"],
        num_proc=data_args.preprocessing_num_workers,
        desc=f"Filtering samples with the same chosen and rejected",
    )
    
    # Cache the processed dataset
    logger.info(f"Caching processed dataset to {cache_path}")
    torch.save(dataset, cache_path)
    
    return dataset

def main():
    
    parser = H4ArgumentParser((ModelArguments, DataArguments, DPOConfig, CurrArgs))
    model_args, data_args, training_args, curr_args = parser.parse()

    # run_name for wandb
    training_args.run_name = f"{model_args.model_name_or_path.split('/')[-1]}-{training_args.seed}-{curr_args.curr_split}" 

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    np.random.seed(training_args.seed)
    set_seed(training_args.seed)
    

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["prompt_id", "messages", "chosen", "rejected", "prompt", "completion", "label"],
    )
    tokenizer = get_tokenizer(model_args, data_args)
    processed_datasets = process_and_cache_dataset(raw_datasets, data_args, tokenizer, "train", training_args)

    # we select half of the raw_dataset as the training set, and the left for evaluation. given the seed 
    num_samples = len(processed_datasets)
    indices = np.random.permutation(num_samples)

    # Split the indices into two halves
    if curr_args.curr_split == "first":
        eval_indices  = indices[:num_samples // 2]
        train_indices = indices[num_samples // 2:]
    elif curr_args.curr_split == "second":
        eval_indices  = indices[num_samples // 2:]
        train_indices = indices[:num_samples // 2]
    else:
        raise ValueError(f"Invalid split: {curr_args.curr_split}")

    # Use the split indices to select the appropriate subsets
    processed_datasets_test = processed_datasets.select(eval_indices)
    processed_datasets_train = processed_datasets.select(train_indices)

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = model_args.model_name_or_path
    if is_adapter_model(model, model_args.model_revision) is True:
        logger.info(f"Loading SFT adapter for {model_args.model_name_or_path=}")
        peft_config = PeftConfig.from_pretrained(model_args.model_name_or_path, revision=model_args.model_revision)
        model_kwargs = dict(
            revision=model_args.base_model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            device_map=get_kbit_device_map() if quantization_config is not None else None,
            quantization_config=quantization_config,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(
            base_model,
            model_args.model_name_or_path,
            revision=model_args.model_revision,
        )
        model_kwargs = None

    ref_model = model
    ref_model_kwargs = model_kwargs

    if model_args.use_peft is True:
        ref_model = None
        ref_model_kwargs = None

    # Calculate total steps and set eval_steps for configurable number of evaluations
    num_training_samples = len(raw_datasets["train"]) // 2 # only half of the training set is used for training
    print(f"num_training_samples: {num_training_samples}, num_devices: {training_args.world_size}, gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    total_steps = (num_training_samples // (training_args.per_device_train_batch_size *  training_args.world_size * training_args.gradient_accumulation_steps)) * training_args.num_train_epochs
    training_args.eval_steps = max(total_steps // curr_args.num_eval, 1)  # Ensure at least 1 step between evaluations
    logger.info(f"Total training steps: {total_steps}, Evaluation every {training_args.eval_steps} steps ({curr_args.num_eval} evaluations)")

    #########################
    trainer = DPODiagnostician(
        model=model,
        ref_model=ref_model,
        model_init_kwargs=model_kwargs,
        ref_model_init_kwargs=ref_model_kwargs,
        args=training_args,
        train_dataset=processed_datasets_train,
        eval_dataset=processed_datasets_test,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
        curr_split=curr_args.curr_split,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    metrics["train_samples"] = len(processed_datasets_train)
    metrics["eval_samples"] = len(processed_datasets_test)
    trainer.log_metrics("train", metrics)
    
    logger.info("*** First round training complete ***")

if __name__ == "__main__":
    main()