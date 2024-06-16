from argparse import ArgumentParser
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from vllm import LLM
from vllm.lora.request import LoRARequest

from leichte_sprache.constants import (
    TEST_ARTICLE,
    LS_USER_PROMPT_TEXT,
    LS_SYSTEM_PROMPT_DICT,
)
from leichte_sprache.utils.model_utils import generate_vllm
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
    args = parser.parse_args()
    return args


def load_peft_model(
    base_modelname: str, peft_modelname: str, merged_path: str = None
) -> tuple[PeftModel, PreTrainedTokenizer]:
    """Load a model trained with parameter-efficient finetuning.

    :param base_modelname: name or path of the full/base model
    :param peft_modelname: name or path of the adapter
    :param merged_path: Optional path to store the merged model, defaults to None
    :return: merged model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(peft_modelname)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_modelname,
        torch_dtype=torch.float16,
        device_map={"": 0},
    )
    base_model.resize_token_embeddings(
        128264
    )  # todo: read this from somewhere (`len(tokenizer)` doesn't work)
    model = PeftModel.from_pretrained(base_model, peft_modelname)
    model.merge_and_unload()
    if merged_path and not os.path.exists(merged_path):
        model.save_pretrained(merged_path)
    elif merged_path:
        logger.warning(
            f"Path {merged_path} already exists! Skipping saving the merged model weights."
        )
    return model, tokenizer


def generate(
    model: PeftModel, tokenizer: PreTrainedTokenizer, messages: list
) -> list[str]:
    """Generate using the finetuned peft model. To do this, apply the chat
    template to the provided messages and then run the generation. Generates
    five sequences for the given prompt. Returns a list of generated texts.

    :param model: finetuned peft model
    :param tokenizer: tokenizer for the finetuned model
    :param messages: messages in the chat format (as the prompt)
    :return: list of generated texts
    """
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        renormalize_logits=True,
        num_return_sequences=5,
    )
    texts = [
        tokenizer.decode(o[input_ids.shape[-1] :], skip_special_tokens=True)
        for o in outputs
    ]
    return texts


def create_prompt(text: str = None) -> list[dict]:
    """Create a prompt in the chat format. If no text input is given,
    use a default example text. In order to evaluate a model finetuned
    to generate Leichte Sprache, use text in standard German as input.
    The system and user prompt used to train the model are used for this
    generation prompt.

    :param text: text that should be transformed to Leichte Sprache. Default: None. In that case, retrieves an example text.
    :return: prompt in the chat format: [{"role": "user", "content": prompt+text}]
    """
    if not text:
        text = TEST_ARTICLE

    messages = [
        LS_SYSTEM_PROMPT_DICT,
        {
            "role": "user",
            "content": LS_USER_PROMPT_TEXT.replace("{text_user}", text),
        },
    ]
    return messages


def run_inference(args: ArgumentParser):
    """Run inference on a finetuned model. Loads the base model and merges the
    finetuned adapter into it. If a path is provided, stores the merged weights
    in the given directory. Retrieves an example text and prompt to translate
    a text into Leichte Sprache and generates five sequences for this text. These
    are logged to the console for manual evaluation.
    """
    messages = create_prompt()

    model, tokenizer = load_peft_model(
        base_modelname=args.base_model,
        peft_modelname=args.peft_model,
        merged_path=args.merged_path,
    )
    texts = generate(model, tokenizer, messages)
    for text in texts:
        logger.info(text + "\n\n #-#-#-#-#-#-#-#-#")
    return


def run_inference_vllm_do_not_use(args: ArgumentParser):
    """Run inference using VLLM. Currently (vllm==0.4.3) doesn't work properly with peft.
    Check again in a couple of releases.
    """
    # todo: fix when vllm has better peft support. Do not use until then.

    messages = create_prompt()
    # model, tokenizer = load_peft_model(
    #     base_modelname=args.base_model,
    #     peft_modelname=args.peft_model,
    #     merged_path=args.merged_path,
    # )
    llm = LLM(
        model=args.base_model,
        # qlora_adapter_name_or_path=args.peft_model,
        max_model_len=1024,
        enable_lora=True,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
    )
    lora_request = LoRARequest(
        "ls_adapter", lora_int_id=1, lora_local_path=args.peft_model
    )
    tokenizer = AutoTokenizer.from_pretrained(args.peft_model)
    outputs = generate_vllm(
        messages=[messages], llm=llm, tokenizer=tokenizer, lora_request=lora_request
    )
    print(outputs)
    return


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
    # run_inference_vllm_do_not_use(args)
