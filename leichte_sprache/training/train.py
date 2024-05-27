import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset, load_dataset
from transformers import HfArgumentParser, TrainingArguments, set_seed
from trl import SFTTrainer

from leichte_sprache.training.utils import create_and_prepare_model


# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    chat_template_format: Optional[str] = field(
        default="none",
        metadata={
            "help": "chatml|zephyr|none. Pass `none` if the dataset is already formatted with the chat template."
        },
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_unsloth: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables UnSloth for training."},
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "The preference dataset to use."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset creating."},
    )
    dataset_text_field: str = field(
        default="text", metadata={"help": "Dataset field to use as input text."}
    )
    max_seq_length: Optional[int] = field(default=512)
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, appends `eos_token_id` at the end of each sample being packed."
        },
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, tokenizers adds special tokens to each sample being packed."
        },
    )
    splits: Optional[str] = field(
        default="train,test",
        metadata={"help": "Comma separate list of the splits to use from the dataset."},
    )


def transform_to_chat(
    row, tokenizer, col_user: str = "translated", col_assistant: str = "text"
):
    # todo docstring
    # todo: change default to final column names
    # todo: move prompts to constants

    text_assistant = row[col_assistant]
    text_user = row[col_user]
    messages = [
        {
            "role": "system",
            "content": "Leichte Sprache hat besondere Regeln. Sätze müssen sehr kurz und verständlich sein. Jeder Satz enthält nur eine Aussage. Es werden nur Aktivsätze verwendet. Sätze bestehen aus den Gliedern Subjekt-Verb-Objekt, z. B. Das Kind streichelt den Hund. Es wird immer das gleiche Wort für die gleiche Sache benutzt. Verneinungen werden, wenn möglich, positiv umformuliert, z. B. 'Das kostet nichts.' zu 'Das ist umsonst'. Der Konjunktiv wird vermieden. Der Genitiv wird durch Fügungen mit 'von' ersetzt, z. B. 'Das Haus des Lehrers' durch 'Das Haus vom Lehrer'. Schwierige Wörter werden erklärt. Zusammengesetzte Wörter werden getrennt, zum Beispiel wird 'Weltall' zu 'Welt-All'. Du bist Übersetzer von Standarddeutsch in Leichte Sprache.",
        },
        {
            "role": "user",
            "content": f"Schreibe den folgenden Text nach den Regeln der Leichten Sprache. Text:\n{text_user}\nText in Leichter Sprache:",
        },
        {"role": "assistant", "content": text_assistant},
    ]
    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # todo constants for column names
    return {"chat": chat}


def prepare_data(dataset_path, tokenizer):
    dataset = load_dataset("csv", data_files=dataset_path, split="train")
    dataset = dataset.map(transform_to_chat, fn_kwargs={"tokenizer": tokenizer})
    return dataset


def run_peft_training(model_args, data_args, training_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # model
    model, peft_config, tokenizer = create_and_prepare_model(
        model_args, data_args, training_args
    )

    # gradient ckpt
    model.config.use_cache = not training_args.gradient_checkpointing
    training_args.gradient_checkpointing = (
        training_args.gradient_checkpointing and not model_args.use_unsloth
    )
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {
            "use_reentrant": model_args.use_reentrant
        }

    # prepare training data
    dataset = prepare_data(data_args.dataset_name, tokenizer)
    dataset = dataset.train_test_split(test_size=0.1)
    train_dataset, eval_dataset = dataset["train"], dataset["test"]

    # trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=data_args.packing,
        dataset_kwargs={
            "append_concat_token": data_args.append_concat_token,
            "add_special_tokens": data_args.add_special_tokens,
        },
        dataset_text_field=data_args.dataset_text_field,
        max_seq_length=data_args.max_seq_length,
    )
    trainer.accelerator.print(f"{trainer.model}")
    trainer.model.print_trainable_parameters()

    # train
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    # saving final model
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()


if __name__ == "__main__":
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1])
        )

    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # todo MLFLow setup
    run_peft_training(model_args, data_args, training_args)