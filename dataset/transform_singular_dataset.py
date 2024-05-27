import os

import datasets
import pandas as pd
import torch
from transformers import (
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
            "content": "Standard-Deutsch richtet sich an erwachsene Muttersprachler, benutzt häufig lange Sätze und einen komplexen Wortschatz mit Fremdwörtern. (Fremd-)Wörter werden nicht erklärt und meist ohne Trennzeichen geschrieben.",
        },
        {
            "role": "user",
            "content": f"Schreibe den folgenden Text für ein gebildetes Publikum um. Halte dich an die im Text genannten Infomationen; Erläuterungen gängiger Begriffe können wegfallen. Text:\n{text}\nText im standardsprachlichen Stil:",
        },
    ]
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
    tmp_dataset_path = "dataset/files/dataset_singular_translated_tmp.csv"
    if not os.path.exists(tmp_dataset_path):
        results, skip_ids = [], []
    else:
        tmp_df = pd.read_csv(tmp_dataset_path)
        results = tmp_df.to_dict("records")
        skip_ids = list(tmp_df.id)

    for row in tqdm(dataset.to_iterable_dataset(), total=len(dataset)):
        if row["id"] in skip_ids:
            continue
        out = pipe(
            row["prompts"],
            max_new_tokens=512,  # todo: configure max new tokens
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            return_full_text=False,
        )
        gen_text = out[0]["generated_text"]
        row.update({"translated": gen_text})
        results.append(row)

        if len(results) % 10 == 0:
            # save the intermediary state
            translation_df = pd.DataFrame(results)
            translation_df.to_csv(tmp_dataset_path, index=False)
    return pd.DataFrame(results)


def transform_singular_dataset():
    # todo: CLI
    dataset_path = "dataset/files/dataset_singular.csv"
    model_id = "DiscoResearch/Llama3_DiscoLM_German_8b_v0.1_experimental"

    pipe = load_pipeline(model_id=model_id)
    dataset = load_dataset(dataset_path=dataset_path, tokenizer=pipe.tokenizer)
    results = generate(dataset=dataset, pipe=pipe)

    print(results)  # todo: remove print
    results.to_csv("dataset/files/dataset_singular_translated.csv")


if __name__ == "__main__":
    transform_singular_dataset()
