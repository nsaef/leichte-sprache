import datasets
import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM

from leichte_sprache.constants import (
    DATASET_SINGULAR_TABLE,
    DATASET_TRANSLATED_TABLE,
    TEXT_COLUMN,
    TRANSLATED_COLUMN,
    PROMPTS_COLUMN,
    ORIG_IDS_COLUMN,
    ID_COLUMN,
)
from leichte_sprache.utils.db_utils import ingest_pandas, get_connector
from leichte_sprache.utils.model_utils import generate_vllm
from leichte_sprache.utils.utils import get_logger


logger = get_logger()


def create_prompt(
    row, text_col_name: str = TEXT_COLUMN, return_col_name: str = PROMPTS_COLUMN
):
    """Create a prompt in the chat message format from the column containing the text.
    Adds a system prompt explaining the linguistic properties of "Standard-Deutsch" and
    a user prompt asking to translate the Leichte Sprache into Standard-Deutsch.
    Returns a dictionary with the return column name as its key and the chat messages list
    as its value.

    :param row: row of a HF dataset
    :param text_col_name: name of the column containing the text to be translated. Default: "text"
    :param return_col_name: name of the new column with the prompt. Default: "prompts"
    :return: dict {return_col_name: chat_messages}
    """

    text = row[text_col_name]
    messages = [
        {
            "role": "system",
            "content": "Standard-Deutsch richtet sich an erwachsene Muttersprachler, benutzt häufig lange Sätze und einen komplexen Wortschatz mit Fremdwörtern. (Fremd-)Wörter werden nicht erklärt und meist ohne Trennzeichen geschrieben.",
        },
        {
            "role": "user",
            "content": f"Schreibe den folgenden Text für ein gebildetes Publikum um. Halte dich an die im Text genannten Infomationen; Erläuterungen gängiger Begriffe fallen weg, Wortbedeutungen werden nicht erklärt. Text:\n{text}\nText im standardsprachlichen Stil:",
        },
    ]
    return {return_col_name: messages}


def load_and_prepare_dataset_from_csv(dataset_path: str) -> datasets.Dataset:
    """Load dataset from a CSV file and create prompts for each row from its "text" column.

    :param dataset_path: path to the CSV file
    :return: HF Dataset based on the CSV file, with added "prompts" column
    """
    dataset = datasets.load_dataset("csv", data_files=dataset_path, split="train")
    dataset = dataset.map(create_prompt)
    dataset = dataset.shuffle()
    return dataset


def load_and_prepare_dataset_from_db() -> datasets.Dataset:
    """Load dataset from a CSV file and create prompts for each row from its "text" column.

    :return: HF Dataset based on the CSV file, with added "prompts" column
    """

    query = f"""SELECT {ID_COLUMN}, {TEXT_COLUMN}, {ORIG_IDS_COLUMN} FROM {DATASET_SINGULAR_TABLE} 
    WHERE id NOT IN (SELECT {ID_COLUMN} FROM {DATASET_TRANSLATED_TABLE});
    """
    conn = get_connector()
    dataset = datasets.Dataset.from_sql(query, con=conn)
    dataset = dataset.map(create_prompt)
    dataset = dataset.shuffle()
    return dataset


def run_vllm_generation(
    dataset: datasets.Dataset,
    model_id: str,
    table_name: str = DATASET_TRANSLATED_TABLE,
    prompt_col_name: str = PROMPTS_COLUMN,
    result_col_name: str = TRANSLATED_COLUMN,
    batch_size: int = 20,
):
    """Generate output from prompts in a dataset using VLLM. The prompts are batched
     according the the `batch_size` parameter. The outputs of each batch are stored
     in a database.

    :param dataset: dataset object
    :param model_id: name or path of the model
    :param table_name: name of the db table in which the results are stored, defaults to "dataset_singular_translated"
    :param prompt_col_name: name of the dataset column containing the prompts, defaults to "prompts"
    :param result_col_name: name of the db table column in which to write the results, defaults to "translated"
    :param batch_size: number of rows per batch, defaults to 20
    """
    llm = LLM(model=model_id, max_model_len=4096, dtype=torch.float16)  # Create an LLM.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    start_idx = 0

    with tqdm(total=len(dataset)) as pbar:
        pbar.set_description("Translating Leichte Sprache to complicated German")
        while start_idx < len(dataset):
            prompts = dataset[prompt_col_name][start_idx : start_idx + batch_size]
            outputs = generate_vllm(prompts, llm, tokenizer, use_tqdm=False)
            texts = [o.outputs[0].text for o in outputs]
            subset = dataset[start_idx : start_idx + batch_size]
            subset[result_col_name] = texts
            translation_df = pd.DataFrame(subset)
            translation_df[prompt_col_name] = translation_df[prompt_col_name].astype(
                str
            )
            ingest_pandas(translation_df, table_name)
            start_idx += batch_size
            pbar.update(batch_size)
    return


def transform_singular_dataset():
    """Transform the singular dataset (consisting only of Leichte Sprache)
    to standard German using an LLM. Save the results in a dedicated DB table.
    By default, intermediary data are ingested into the DB every 5 steps.
    Only rows that have not yet been translated are retrieved from the singular
    dataset table.
    """
    logger.info("Translating Leichte Sprache into standard German...")

    model_id = "DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1-4bit-awq"
    dataset = load_and_prepare_dataset_from_db()
    run_vllm_generation(dataset, model_id)
    return


if __name__ == "__main__":
    transform_singular_dataset()
