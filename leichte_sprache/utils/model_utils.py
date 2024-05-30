import datasets
import pandas as pd
import torch
from transformers import (
    pipeline,
    Pipeline,
)
import tiktoken

from tqdm.auto import tqdm

from leichte_sprache.utils.db_utils import ingest_pandas


def load_pipeline(model_id: str) -> Pipeline:
    # todo: docstring

    #  model_kwargs={"load_in_8bit": True}
    pipe = pipeline(
        "text-generation",
        model=model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    return pipe


def generate(
    dataset: datasets.Dataset,
    pipe: Pipeline,
    save_intermediary_steps: bool = True,
    table_name: str = "dataset_singular_translated",
    prompt_col_name: str = "prompts",
    result_col_name: str = "translated",
) -> pd.DataFrame:
    # todo: docstring, return value typing
    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    results = []

    for row in tqdm(dataset.to_iterable_dataset(), total=len(dataset)):
        out = pipe(
            row[prompt_col_name],
            max_new_tokens=512,  # todo: configure max new tokens
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            return_full_text=False,
        )
        gen_text = out[0]["generated_text"]
        row.update({result_col_name: gen_text})
        results.append(row)

        if len(results) % 5 == 0 and save_intermediary_steps:
            # save the intermediary state
            translation_df = pd.DataFrame(results)
            translation_df[prompt_col_name] = translation_df[prompt_col_name].astype(
                str
            )
            ingest_pandas(translation_df, table_name)
            results = []
    return pd.DataFrame(results)


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
