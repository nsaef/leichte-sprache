import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from leichte_sprache.constants import (
    TEST_ARTICLE,
    LS_USER_PROMPT_TEXT,
    LS_SYSTEM_PROMPT_DICT,
)


def load_model(base_modelname, peft_modelname):
    tokenizer = AutoTokenizer.from_pretrained(peft_modelname)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_modelname,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        # use_flash_attention_2=True, #todo: upgrade cuda to 11.6 to use flash attention 2
    )
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, peft_modelname)
    return model, tokenizer


def generate(model, tokenizer: PreTrainedTokenizer, messages: list):
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
        # return_full_text=False,
        # skip_special_tokens=True,
    )
    response = outputs[0][input_ids.shape[-1] :]
    output = tokenizer.decode(response, skip_special_tokens=True)
    return output


def create_example_prompts(text: str = None):
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
    base_modelname = "DiscoResearch/Llama3_DiscoLM_German_8b_v0.1_experimental"
    peft_modelname = "/home/nasrin/data/trainings/test-sft-lora-unsloth"

    messages = create_example_prompts()
    model, tokenizer = load_model(
        base_modelname=base_modelname, peft_modelname=peft_modelname
    )
    result = generate(model, tokenizer, messages)
    print(result)


if __name__ == "__main__":
    run_inference()
