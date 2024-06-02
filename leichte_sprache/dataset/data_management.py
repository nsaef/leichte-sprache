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


def setup_crawler_db_table():
    """Initial setup of an SQLite table for storing the crawled data.
    Creates the columns `source`, `text`, `url`, `crawl_timestamp`, `title`, `release_date`, `full_text` and `id`
    """
    # todo: constants for column names
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
    return


def calculate_rouge(predictions: list[str], references: list[str]) -> list[float]:
    # todo: docstrings
    rouge = evaluate.load("rouge")
    scores = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=["rouge2"],
        use_aggregator=False,
    )
    return scores["rouge2"]


def recognize_language(texts: list[str]):
    languages = [Language.ENGLISH, Language.GERMAN]
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    langs = detector.detect_languages_in_parallel_of(texts)
    lang_strings = [l.name for l in langs]
    return lang_strings


def filter_translated_dataset(dataset: Dataset) -> Dataset:
    # rouge2
    rouge2_scores = calculate_rouge(dataset[LS_COLUMN], dataset[SG_COLUMN])
    dataset = dataset.add_column("rouge2", rouge2_scores)

    # lang recognition
    detected_langs = recognize_language(dataset[SG_COLUMN])
    dataset = dataset.add_column("lang_sg", detected_langs)

    dataset_filtered = dataset.filter(
        lambda x: x["rouge2"] < 0.7 and x["lang_sg"] == Language.GERMAN.name
    )
    diff = len(dataset) - len(dataset_filtered)
    print(
        f"Removed {diff} rows due to language mismatches or similarity ({diff/len(dataset)*100} %)"
    )
    return dataset_filtered


def create_hf_dataset():
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
    return


if __name__ == "__main__":
    # todo add argparse
    initialize_crawler_table = False
    transform_crawler_to_singular = False
    create_dataset = True
    # todo: create commandline command (setup)

    if initialize_crawler_table:
        setup_crawler_db_table()

    if transform_crawler_to_singular:
        transform_to_singular_dataset()

    if create_dataset:
        create_hf_dataset()
