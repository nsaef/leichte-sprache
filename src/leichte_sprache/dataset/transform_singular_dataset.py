import datasets

from leichte_sprache.utils.db_utils import ingest_pandas, get_connector
from leichte_sprache.utils.model_utils import load_pipeline, generate


def create_prompt(row, text_col_name: str = "text", return_col_name: str = "prompts"):
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
            "content": f"Schreibe den folgenden Text für ein gebildetes Publikum um. Halte dich an die im Text genannten Infomationen; Erläuterungen gängiger Begriffe können wegfallen. Text:\n{text}\nText im standardsprachlichen Stil:",
        },
    ]
    return {return_col_name: messages}


def load_and_prepare_dataset(dataset_path: str) -> datasets.Dataset:
    """Load dataset from a CSV file and create prompts for each row from its "text" column.

    :param dataset_path: path to the CSV file
    :return: HF Dataset based on the CSV file, with added "prompts" column
    """
    dataset = datasets.load_dataset("csv", data_files=dataset_path, split="train")
    dataset = dataset.map(create_prompt)
    return dataset


def load_and_prepare_dataset_from_db() -> datasets.Dataset:
    """Load dataset from a CSV file and create prompts for each row from its "text" column.

    :return: HF Dataset based on the CSV file, with added "prompts" column
    """

    query = """SELECT id, text, orig_ids FROM dataset_singular 
    WHERE id NOT IN (SELECT id FROM dataset_singular_translated);
    """
    conn = get_connector()
    dataset = datasets.Dataset.from_sql(query, con=conn)
    dataset = dataset.map(create_prompt)
    return dataset


def transform_singular_dataset():
    print("Translating Leichte Sprache into standard German...")

    # model_id = "DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1"
    model_id = "DiscoResearch/Llama3-DiscoLeo-Instruct-8B-v0.1-4bit-awq"
    pipe = load_pipeline(model_id=model_id)
    dataset = load_and_prepare_dataset_from_db()
    results = generate(dataset=dataset, pipe=pipe)
    ingest_pandas(results, "dataset_singular_translated")
