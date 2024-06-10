from transformers import (
    PreTrainedTokenizer,
)
import tiktoken
from vllm import LLM, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest


def generate_vllm(
    messages: list,
    llm: LLM,
    tokenizer: PreTrainedTokenizer,
    use_tqdm: bool = True,
    lora_request: LoRARequest = None,
) -> list[RequestOutput]:
    """Generate with a transformers-compatible model using VLLM. Messages are
    converted to plain text via the chat template and then passed to the model.

    :param messages: list of prompts in the chat messages format
    :param llm: VLLM LLAM instance
    :param tokenizer: the model's tokenizer (to apply the correct chat template)
    :param use_tqdm: show a tqdm progress bar. Default: True
    :param lora_request: Optional VLLM LoRARequest instance. Default: None
    :return: list of VLLM RequestOutput objects
    """
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    sampling_params = SamplingParams(
        max_tokens=512,
        stop_token_ids=terminators,
        temperature=0.6,
        top_p=0.9,
        skip_special_tokens=True,
        top_k=50,
    )
    prompts = [
        tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False
        )
        for message in messages
    ]
    outputs = llm.generate(
        prompts, sampling_params, use_tqdm=use_tqdm, lora_request=lora_request
    )
    return outputs


def count_tokens_openai(model: str, texts: list[str]) -> list[int]:
    """Count tokens for a batch of texts using the OpenAI library tiktoken.

    :param model: OpenAI modelname
    :param texts: list of texts
    :return: list with the number of tokens of each text
    """
    encoding = tiktoken.encoding_for_model(model)
    tokenized = encoding.encode_batch(texts)
    n_tokens = [len(t) for t in tokenized]
    return n_tokens


def count_tokens_from_messages_openai(model: str, messages: list[dict]):
    """Return the number of tokens used by a list of messages.

    :param model: OpenAI modelname
    :param messages: list of dicst in the chat message format
    :return: number of tokens in the messages
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
        )
        return count_tokens_from_messages_openai(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
        )
        return count_tokens_from_messages_openai(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
