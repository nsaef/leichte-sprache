import os

import datasets
import pandas as pd
import torch
from transformers import (
    pipeline,
    Pipeline,
    PreTrainedTokenizer,
)
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
