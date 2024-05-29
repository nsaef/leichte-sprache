import datasets

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


def transform_singular_dataset():
    # todo: CLI
    # todo: load from database
    dataset_path = "data/datasets/dataset_singular.csv"
    model_id = "DiscoResearch/Llama3_DiscoLM_German_8b_v0.1_experimental"

    pipe = load_pipeline(model_id=model_id)
    dataset = load_and_prepare_dataset(dataset_path=dataset_path)
    results = generate(dataset=dataset, pipe=pipe)

    print(results)  # todo: remove print
    # todo: save to database
    results.to_csv("data/datasets/dataset_singular_translated.csv")


if __name__ == "__main__":
    transform_singular_dataset()
