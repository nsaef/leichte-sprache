import os

from datasets import Dataset
import evaluate
from lingua import Language, LanguageDetectorBuilder
import pandas as pd

from leichte_sprache.constants import (
    CRAWLER_TABLE,
    DATASET_SINGULAR_TABLE,
    DATASET_TRANSLATED_TABLE,
    HF_DATASET_NAME,
    LS_COLUMN,
    SG_COLUMN,
    SRC_COLUMN,
    ID_COLUMN,
    URL_COLUMN,
    CRAWL_TS_COLUMN,
    TEXT_COLUMN,
    TITLE_COLUMN,
    RELEASE_COLUMN,
    FULL_TEXT_COLUMN,
    TRANSLATED_COLUMN,
)
from leichte_sprache.utils.db_utils import (
    create_column_dict,
    create_table,
    get_connector,
    ingest_pandas,
)
from leichte_sprache.utils.utils import get_logger


logger = get_logger()


def setup_crawler_db_table():
    """Initial setup of an SQLite table for storing the crawled data.
    Creates the columns `source`, `text`, `url`, `crawl_timestamp`, `title`, `release_date`, `full_text` and `id`
    """
    logger.info(f"Initializing table {CRAWLER_TABLE}")
    columns = [
        create_column_dict(
            col_name=SRC_COLUMN, col_dtype="varchar(124)", not_null=True
        ),
        create_column_dict(col_name=TEXT_COLUMN, col_dtype="text", not_null=True),
        create_column_dict(col_name=URL_COLUMN, col_dtype="text", not_null=True),
        create_column_dict(
            col_name=CRAWL_TS_COLUMN, col_dtype="datetime", not_null=True
        ),
        create_column_dict(col_name=TITLE_COLUMN, col_dtype="varchar(256)"),
        create_column_dict(col_name=RELEASE_COLUMN, col_dtype="datetime"),
        create_column_dict(col_name=FULL_TEXT_COLUMN, col_dtype="text"),
        create_column_dict(col_name=ID_COLUMN, col_dtype="varchar(124)", not_null=True),
    ]
    create_table(CRAWLER_TABLE, columns=columns)
    return


def transform_to_singular_dataset():
    """Write the data in the crawler table into a new table ready for enrichment."""
    logger.info(
        f"Transforming data from the table {CRAWLER_TABLE} into a singular dataset..."
    )
    conn = get_connector()
    df = pd.read_sql(f"SELECT * FROM {CRAWLER_TABLE}", con=conn)

    # create new df compatible with the singular dataset table
    data_df = df[[ID_COLUMN, FULL_TEXT_COLUMN, URL_COLUMN]]
    data_df = data_df.rename(
        columns={FULL_TEXT_COLUMN: TEXT_COLUMN, URL_COLUMN: "orig_ids"}
    )

    # load singular dataset, combine with new data, drop all duplicates
    singular_df = pd.read_sql(f"SELECT * FROM {DATASET_SINGULAR_TABLE}", con=conn)
    complete_df = pd.concat([singular_df, data_df])
    complete_df = complete_df.drop_duplicates(subset="id")

    # drop the old table and re-add the complete data
    ingest_pandas(complete_df, DATASET_SINGULAR_TABLE, if_exists="replace")
    logger.info(f"Stored {len(complete_df)} rows in table {DATASET_SINGULAR_TABLE}")
    return


def calculate_rouge(predictions: list[str], references: list[str]) -> list[float]:
    """Calculate rouge2 between two lists of texts.

    :param predictions: list of predicted texts
    :param references: list of reference texts
    :return: list of rouge2 scores
    """
    rouge = evaluate.load("rouge")
    scores = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rouge2"],
        use_aggregator=False,
    )
    return scores["rouge2"]


def recognize_language(texts: list[str]) -> list[str]:
    """Run automated language recognition on a list of texts, such as  `GERMAN` or `ENGLISH`.

    :param texts: list of texts
    :return: list of language names
    """
    languages = [Language.ENGLISH, Language.GERMAN]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    langs = detector.detect_languages_in_parallel_of(texts)
    lang_strings = [l.name if l else None for l in langs]
    return lang_strings


def filter_translated_dataset(
    dataset: Dataset,
    rouge_threshold: float = 0.7,
    ls_col: str = LS_COLUMN,
    sg_col: str = SG_COLUMN,
) -> Dataset:
    """
    Filter the automatically translated dataset (Leichte Sprache to complicated German)
    to remove bad examples. The following filters are run:
    - bigram overlap (rouge2): remove examples above a certain rouge threshold,
      to filter examples that were barely altered
    - language recognition: remove examples that were wrongfully generated in English

    :param dataset: dataset containing texts in Leichte Sprache and automatically translated standard German
    :param rouge_threshold: maximum allowed rouge2 value
    :param ls_col: name of the column containing the texts in Leichte Sprache. Default: `leichte_sprache`
    :param sg_col: name of the column containing the texts in standard German. Default: `standard_german`
    :return: dataset without the bad examples
    """
    # rouge2
    rouge2_scores = calculate_rouge(dataset[ls_col], dataset[sg_col])
    dataset = dataset.add_column("rouge2", rouge2_scores)

    # lang recognition
    detected_langs = recognize_language(dataset[sg_col])
    dataset = dataset.add_column("lang_sg", detected_langs)

    dataset_filtered = dataset.filter(
        lambda x: x["rouge2"] < rouge_threshold and x["lang_sg"] == Language.GERMAN.name
    )
    diff = len(dataset) - len(dataset_filtered)
    logger.info(
        f"Removed {diff} rows due to language mismatches or similarity ({diff/len(dataset)*100} %)"
    )
    return dataset_filtered


def remove_bad_generations():
    """Remove bad generations from the database to prevent training on subpar data.
    The rows containing subpar translations are removed from the table for the
    translated dataset (so they are retrieved again for the next translation run).

     The following filters are applied:
    - bigram overlap (rouge2): remove examples above a certain rouge threshold,
      to filter examples that were barely altered
    - language recognition: remove examples that were wrongfully generated in English
    """
    logger.info("Removing bad generations from the translated dataset...")

    conn = get_connector()
    sql = f"SELECT * FROM {DATASET_TRANSLATED_TABLE}"
    dataset = Dataset.from_sql(sql, con=conn)
    dataset_filtered = filter_translated_dataset(
        dataset, ls_col="text", sg_col="translated"
    )
    dataset_filtered = dataset_filtered.remove_columns(["rouge2", "lang_sg"])

    # remove duplicates
    df = dataset_filtered.to_pandas()
    df = df.drop_duplicates(subset="id")
    logger.info(f"Removed {len(dataset_filtered) - len(df)} duplicate rows")

    ingest_pandas(df, DATASET_TRANSLATED_TABLE, if_exists="replace")
    logger.info("Replaced table with filtered dataset")
    return


def create_hf_dataset():
    """
    Create a HuggingFace dataset from the translated data and corresponding metadata.
    Filter out bad examples and push the dataset to the Hub.
    """
    logger.info("Creating HF dataset...")

    # load data from table
    conn = get_connector()
    sql = f"""
        SELECT t.{ID_COLUMN}, t.{TEXT_COLUMN}, t.{TRANSLATED_COLUMN}, c.{SRC_COLUMN}, c.{URL_COLUMN}, c.{RELEASE_COLUMN} 
        FROM {DATASET_TRANSLATED_TABLE} t
        JOIN {CRAWLER_TABLE} c ON t.{ID_COLUMN}=c.{ID_COLUMN}
    """
    dataset = Dataset.from_sql(sql, con=conn)
    dataset = dataset.rename_columns(
        {TEXT_COLUMN: LS_COLUMN, TRANSLATED_COLUMN: SG_COLUMN}
    )

    filtered_df = filter_translated_dataset(dataset)

    # shuffle
    filtered_df = filtered_df.shuffle()

    # push to hub
    dataset.push_to_hub(HF_DATASET_NAME, token=os.getenv("HF_TOKEN"))
    logger.info(f"Pushed dataset with {len(dataset)} lines to repo {HF_DATASET_NAME}")
    return


# todo: different subsets of the dataset for generation and classification
# docs:
# english_dataset.push_to_hub("<organization>/<dataset_id>", "en")
# french_dataset.push_to_hub("<organization>/<dataset_id>", "fr")
# # later
# english_dataset = load_dataset("<organization>/<dataset_id>", "en")
# french_dataset = load_dataset("<organization>/<dataset_id>", "fr")


# if __name__ == "__main__":
#     pass
