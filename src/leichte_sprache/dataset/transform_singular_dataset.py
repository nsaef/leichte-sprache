import datasets
import pandas as pd
from vllm import SamplingParams

from leichte_sprache.constants import (
    DATASET_SINGULAR_TABLE,
    DATASET_TRANSLATED_TABLE,
    TEXT_COLUMN,
    TRANSLATED_COLUMN,
    PROMPTS_COLUMN,
    ORIG_IDS_COLUMN,
    ID_COLUMN,
)
from leichte_sprache.utils.db_utils import get_connector
from leichte_sprache.utils.model_utils import run_vllm_batch_generation
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


def process_vllm_outputs(
    outputs: list,
    start_idx: int,
    end_idx: int,
    result_col_name: str,
    prompt_col_name: str,
    dataset: datasets.Dataset,
) -> pd.DataFrame:
    """Process the vLLM outputs and create a DataFrame that can be inserted
    into the project DB table. Prompts are converted from list of dictionaries
    to strings.

    :param outputs: list of vLLM RequestOutput objects
    :param start_idx: start index of the current batch
    :param end_idx: end index of the current batch
    :param result_col_name: name of the DB column to store the generation results in
    :param prompt_col_name: name of the DB column to store the prompts in
    :param dataset: full dataset
    :return: dataframe generation results and metadata
    """
    texts = [o.outputs[0].text for o in outputs]
    subset = dataset[start_idx:end_idx]
    subset[result_col_name] = texts
    translation_df = pd.DataFrame(subset)
    translation_df[prompt_col_name] = translation_df[prompt_col_name].astype(str)
    return translation_df


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
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        skip_special_tokens=True,
        top_k=50,
        n=1,
    )
    run_vllm_batch_generation(
        dataset=dataset,
        model_id=model_id,
        result_table_name=DATASET_TRANSLATED_TABLE,
        ds_prompt_col_name=PROMPTS_COLUMN,
        process_output=process_vllm_outputs,
        output_fn_kwargs={
            "result_col_name": TRANSLATED_COLUMN,
            "prompt_col_name": PROMPTS_COLUMN,
            "dataset": dataset,
        },
        batch_size=20,
        max_model_len=4096,
        sampling_params=sampling_params,
    )
    return


if __name__ == "__main__":
    transform_singular_dataset()
