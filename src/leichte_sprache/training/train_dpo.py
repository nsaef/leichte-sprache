# flake8: noqa
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
"""
# regular:
python examples/scripts/dpo.py \
    --dataset_name=trl-internal-testing/hh-rlhf-helpful-base-trl-style \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

# peft:
python examples/scripts/dpo.py \
    --dataset_name=trl-internal-testing/hh-rlhf-helpful-base-trl-style \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --learning_rate 1e-3 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="dpo_anthropic_hh" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""
import sys
import logging
import os
from contextlib import nullcontext

TRL_USE_RICH = os.environ.get("TRL_USE_RICH", False)

from trl.commands.cli_utils import DPOScriptArguments, init_zero_verbose, TrlParser

from leichte_sprache.utils.training_utils import transform_to_chat


if TRL_USE_RICH:
    init_zero_verbose()
    FORMAT = "%(message)s"

    from rich.console import Console
    from rich.logging import RichHandler

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    RichProgressCallback,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from unsloth import FastLanguageModel


if TRL_USE_RICH:
    logging.basicConfig(
        format=FORMAT, datefmt="[%X]", handlers=[RichHandler()], level=logging.INFO
    )


def run_dpo_training(args, training_args, model_config):
    os.environ["MLFLOW_EXPERIMENT_NAME"] = os.getenv("MLFLOW_EXPERIMENT_NAME") + "_dpo"

    # Force use our print callback
    if TRL_USE_RICH:
        training_args.disable_tqdm = True
        console = Console()

    ################
    # Model & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
    )

    max_seq_length = 512
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.model_name_or_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        **model_kwargs,
    )

    # Do model patching and add fast LoRA weights
    peft_config = get_peft_config(model_config)
    if isinstance(peft_config.target_modules, str):
        peft_config.target_modules = peft_config.target_modules.split(",")
    model = FastLanguageModel.get_peft_model(
        model,
        r=peft_config.r,
        target_modules=peft_config.target_modules,
        lora_alpha=peft_config.lora_alpha,
        lora_dropout=peft_config.lora_dropout,  # Dropout = 0 is currently optimized
        bias=peft_config.bias,  # Bias = "none" is currently optimized
        use_gradient_checkpointing=True,
        random_state=4711,
    )
    if peft_config is None:
        model_ref = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, **model_kwargs
        )
    else:
        model_ref = None
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    ################
    # Optional rich context managers
    ###############
    init_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status("[bold green]Initializing the DPOTrainer...")
    )
    save_context = (
        nullcontext()
        if not TRL_USE_RICH
        else console.status(
            f"[bold green]Training completed! Saving the model to {training_args.output_dir}"
        )
    )

    ################
    # Dataset
    ################
    ds = load_dataset(args.dataset_name)
    if args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(50))

    ds = ds.map(
        transform_to_chat,
        fn_kwargs={
            "tokenizer": tokenizer,
            "col_assistant": "chosen",
            "col_return_value": "chosen",
            "system_msg": False,
            "user_msg": False,
        },
    )
    ds = ds.map(
        transform_to_chat,
        fn_kwargs={
            "tokenizer": tokenizer,
            "col_assistant": "rejected",
            "col_return_value": "rejected",
            "system_msg": False,
            "user_msg": False,
        },
    )
    ds = ds["train"].train_test_split(test_size=0.1)
    train_dataset = ds[args.dataset_train_split]
    eval_dataset = ds[args.dataset_test_split]

    ################
    # Training
    ################
    with init_context:
        trainer = DPOTrainer(
            model,
            model_ref,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config,
            callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
        )

    trainer.train()

    with save_context:
        trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args, training_args, model_config = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args, training_args, model_config = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1])
        )
    else:
        args, training_args, model_config = parser.parse_args_and_config()

    run_dpo_training(args, training_args, model_config)
