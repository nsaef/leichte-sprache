from argparse import ArgumentParser
import os

from awq import AutoAWQForCausalLM
from peft import PeftModel
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)

from leichte_sprache.utils.utils import get_logger


logger = get_logger()


def parse_args() -> ArgumentParser:
    """Parse the command line arguments to select the sources to crawl.

    :return: ArgumentParser with command line arguments.
    """

    parser = ArgumentParser(
        prog="Leichte Sprache Inference",
        description="Run inference on a finetuned model",
    )
    parser.add_argument(
        "--base_model",
        required=True,
        help="Name or path of the base model of the finetuned model",
    )
    parser.add_argument(
        "--peft_model",
        required=True,
        help="Name or path of the finetuned model weights",
    )
    parser.add_argument(
        "--merged_path",
        help="Optional path to store the merged model weights.",
    )
    parser.add_argument(
        "--quantized_path",
        help="Optional path to store the merged model weights.",
    )
    args = parser.parse_args()
    return args


def merge_peft_model(
    base_modelname: str, peft_modelname: str, merged_path: str
) -> tuple[PeftModel, PreTrainedTokenizer]:
    """Load a model trained with parameter-efficient finetuning.

    :param base_modelname: name or path of the full/base model
    :param peft_modelname: name or path of the adapter
    :param merged_path: path to store the merged model
    """
    tokenizer = AutoTokenizer.from_pretrained(peft_modelname)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_modelname,
        torch_dtype=torch.float16,
        device_map={"": 0},
        attn_implementation="flash_attention_2",
    )
    base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    model = PeftModel.from_pretrained(base_model, peft_modelname)
    model = model.merge_and_unload()
    if merged_path and not os.path.exists(merged_path):
        model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
    elif merged_path:
        logger.warning(
            f"Path {merged_path} already exists! Skipping saving the merged model weights."
        )
    return


def run_quantization(model_path: str, quant_path: str):
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": 4,
        "version": "GEMM",
    }

    # Load model
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, **{"low_cpu_mem_usage": True}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Quantize
    model.quantize(tokenizer, quant_config=quant_config)

    # Save quantized model
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    return


def quantize_model(args):
    """Load the base model and the adapter and merge the adapter into the base model.
    Save the merged model under a given path and pass that path to the quantization.
    Currently uses AWQ for quantization.

    :param args: _description_
    """
    merge_peft_model(
        base_modelname=args.base_model,
        peft_modelname=args.peft_model,
        merged_path=args.merged_path,
    )
    run_quantization(args.merged_path, args.quantized_path)


if __name__ == "__main__":
    args = parse_args()
    quantize_model(args)
