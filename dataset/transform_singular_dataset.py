import datasets
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    Pipeline,
    PreTrainedTokenizer,
)
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm


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


def create_prompt(row, tokenizer: PreTrainedTokenizer):
    # todo docstring
    text = row["text"]
    messages = [
        {
            "role": "system",
            "content": "Du bist ein Übersetzer für Leichte Sprache. Du übersetzt Leichte Sprache zu Standard-Deutsch. Standard-Deutsch benutzt längere Sätze und einen komplexeren Wortschatz als Leichte Sprache. Dafür können Sätze miteinander verbunden und neu strukturiert werden.",
        },
        {
            "role": "user",
            "content": f"Formulier den folgenden Text um, so dass er den Regeln von Standard-Deutsch folgt:\n{text}\nText auf Standard-Deutsch:",
        },
    ]
    # chats = tokenizer.apply_chat_template(messages, tokenize=False)
    # todo constants for column names
    return {"prompts": messages}


def load_dataset(dataset_path: str, tokenizer: PreTrainedTokenizer) -> datasets.Dataset:
    # todo: docstring
    dataset = datasets.load_dataset("csv", data_files=dataset_path, split="train")
    dataset = dataset.map(create_prompt, fn_kwargs={"tokenizer": tokenizer})
    return dataset


def generate(dataset: datasets.Dataset, pipe: Pipeline):
    # todo: docstring, return value typing
    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]
    results = []

    for prompt in tqdm(dataset["prompts"]):
        out = pipe(
            prompt,
            max_new_tokens=256,  # todo: configure max new tokens
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            return_full_text=False,
        )
        results.append(out)

    """    
    for out in tqdm(pipe(
        KeyDataset(dataset, "prompts"),
        max_new_tokens=256, #todo: configure max new tokens
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        return_full_text=False,
        )):
        results.append(out)
        """
    generated_texts = [res[0]["generated_text"] for res in results]
    return generated_texts


def transform_singular_dataset():
    # todo: CLI
    dataset_path = "dataset/files/dataset_singular.csv"
    model_id = "DiscoResearch/Llama3_DiscoLM_German_8b_v0.1_experimental"

    pipe = load_pipeline(model_id=model_id)
    dataset = load_dataset(dataset_path=dataset_path, tokenizer=pipe.tokenizer)
    results = generate(dataset=dataset, pipe=pipe)

    print(results)


if __name__ == "__main__":
    transform_singular_dataset()
