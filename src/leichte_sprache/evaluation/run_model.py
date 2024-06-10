import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from leichte_sprache.constants import (
    TEST_ARTICLE,
    LS_USER_PROMPT_TEXT,
    LS_SYSTEM_PROMPT_DICT,
)


def load_peft_model(base_modelname: str, peft_modelname: str, merged_path: str = None):
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
    if merged_path:
        model.save_pretrained(merged_path)
    return model, tokenizer


def generate(model, tokenizer: PreTrainedTokenizer, messages: list):
    # todo docstring
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


def create_example_prompts(text: str = None):
    """_summary_
    # todo docstring, improve
    :param text: _description_, defaults to None
    :return: _description_
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


def run_inference():
    # todo docstring, argparse
    base_modelname = "DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1"
    peft_modelname = "/home/nasrin/data/trainings/discolm-ls-01/checkpoint-500"
    merged_path = "/home/nasrin/data/trainings/discolm-ls-01/merged"
    messages = create_example_prompts()

    model, tokenizer = load_peft_model(
        base_modelname=base_modelname,
        peft_modelname=peft_modelname,
        merged_path=merged_path,
    )
    texts = generate(model, tokenizer, messages)
    for text in texts:
        print(text + "\n\n #-#-#-#-#-#-#-#-#")
    return

    # todo: remove or fix when vllm has better peft support
    # if not os.path.exists(merged_path):
    #     model, tokenizer = merge_peft_model(
    #         base_modelname=base_modelname, peft_modelname=peft_modelname, new_path=merged_path
    #     )
    # else:
    #     tokenizer = AutoTokenizer.from_pretrained(peft_modelname)

    # llm = LLM(model=base_modelname, max_model_len=1024, enable_lora=True)
    # lora_request = LoRARequest("ls_adapter", lora_int_id=1, lora_local_path=merged_path)
    # outputs = generate_vllm(messages=messages, llm=llm, tokenizer=tokenizer, lora_request=lora_request)
    # print(outputs)


if __name__ == "__main__":
    run_inference()
