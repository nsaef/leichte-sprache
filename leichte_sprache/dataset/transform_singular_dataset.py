import datasets
from transformers import (
    PreTrainedTokenizer,
)

from leichte_sprache.utils.model_utils import load_pipeline, generate


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


def transform_singular_dataset():
    # todo: CLI
    dataset_path = "data/datasets/dataset_singular.csv"
    model_id = "DiscoResearch/Llama3_DiscoLM_German_8b_v0.1_experimental"

    pipe = load_pipeline(model_id=model_id)
    dataset = load_dataset(dataset_path=dataset_path, tokenizer=pipe.tokenizer)
    results = generate(dataset=dataset, pipe=pipe)

    print(results)  # todo: remove print
    results.to_csv("data/datasets/dataset_singular_translated.csv")


if __name__ == "__main__":
    transform_singular_dataset()
