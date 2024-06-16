import os
import uuid

from datasets import (
    Dataset,
    load_dataset,
    ClassLabel,
    Features,
    Value,
    concatenate_datasets,
)
from lingua import Language
import pandas as pd

from leichte_sprache.constants import (
    CRAWLER_TABLE,
    DATASET_SINGULAR_TABLE,
    DATASET_TRANSLATED_TABLE,
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
    LS_LABEL,
    SG_LABEL,
)
from leichte_sprache.dataset.transform_singular_dataset import (
    transform_singular_dataset,
)
from leichte_sprache.dataset.crawler import run_crawler
from leichte_sprache.evaluation.score import calculate_rouge, recognize_language
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
    Creates the columns `source`, `text`, `url`, `crawl_timestamp`,
    `title`, `release_date`, `full_text` and `id` if the table doesn't
    already exist
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


def filter_translated_dataset(
    dataset: Dataset,
    rouge_threshold: float = 0.65,
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
    push_to_hf_hub(dataset, os.getenv("HF_DATASET_NAME"))
    return


def push_to_hf_hub(dataset: Dataset, name: str):
    """Push a dataset to the huggingface data hub.

    :param dataset: dataset object
    :param name: dtaset name
    """
    dataset.push_to_hub(name, token=os.getenv("HF_TOKEN"))
    logger.info(f"Pushed dataset with {len(dataset)} lines to repo {name}")
    return


def load_konvens(dirname: str) -> pd.DataFrame:
    """
    Load the konvens dataset file(s) and parse them into a standardized format. Combine sentences that belong to the same text
    and preserve the original IDs. Returns a dataframe containing a UUID, the combined text and the original IDs.
    :param dirname: path of the directory containing the konvens file(s) relative to the root directory
    :return: dataframe contianing the konvens data
    """
    filenames = os.listdir(dirname)
    data = []
    for fname in filenames:
        if not fname.startswith("konvens_"):
            continue
        fpath = f"{dirname}/{fname}"
        df = pd.read_csv(fpath)
        topics = set(df.topic)
        for topic in topics:
            text = "\n".join(list(df[df.topic == topic].phrase)).replace(
                " \\newline ", "\n"
            )
            orig_ids = ",".join(list(df[df.topic == topic]["sent-id"].astype(str)))
            data.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "orig_ids": orig_ids,
                }
            )
    data_df = pd.DataFrame(data)
    return data_df


def create_singular_dataset():
    """Create the singular dataset (only Leichte Sprache) by loading
    the Konvens data and all crawled data and storing the relevant data
    in a new combined database table.
    """
    dirname = "data/datasets"
    datasets = []

    konvens_data = load_konvens(dirname)
    datasets.append(konvens_data)

    # load the konvens data into DB
    df = pd.concat(datasets) if len(datasets) > 1 else datasets[0]
    ingest_pandas(df, DATASET_SINGULAR_TABLE)

    # store the relevant part of the crawled texts in the singular dataset table
    transform_to_singular_dataset()
    return


def crawl_all_sources():
    """Crawl all available data sources and store them in the DB table `crawled_texts`."""
    run_crawler(dlf_dict=True, dlf_news=True, ndr=True, mdr_dict=True, mdr_news=True)
    return


def run_data_pipeline():
    """Run the complete data pipeline. Create the crawler table if it doesn't exist
    already, crawl all available datasets, load the konvens data and combine all
    available data in Leichte Sprache in a new database table.
    Take the complete content of that table and translate it to standard German
    using an LLM. Filter the automatically created data and create a dataset with
    all rows that match the filter criteria. Push this dataset to the HF hub.

    Take care! Running this whole workflow will take some hours (depending on your hardware).
    """
    setup_crawler_db_table()
    run_crawler(dlf_dict=True, dlf_news=True, ndr=True, mdr_dict=True, mdr_news=True)
    create_singular_dataset()
    transform_singular_dataset()
    create_hf_dataset()
    return


def create_classification_dataset():
    ls_dataset = load_dataset(os.getenv("HF_DATASET_NAME"), split="train")

    # load a news dataset, as this is the most common text type in the LS dataset
    news_dataset = load_dataset("bjoernp/tagesschau-010124-020524", split="train").map(
        lambda x: {"content": f"{x['headline']}\n{x['article']}"}
    )

    target_len = int((len(ls_dataset) - len(news_dataset)) / 2)
    wiki_dataset = (
        load_dataset("wikimedia/wikipedia", "20231101.de", split="train")
        .shuffle()
        .select(range(target_len))
        .map(lambda x: {"content": f"{x['title']}\n{x['text']}"})
    )
    web_dataset = (
        load_dataset("uonlp/CulturaX", "de", split="train", streaming=True)
        .shuffle()
        .take(target_len)
    )
    web_texts = [row["text"] for row in web_dataset]

    ls = Dataset.from_dict(
        {
            "text": ls_dataset[LS_COLUMN],
            "label": [LS_LABEL] * len(ls_dataset),
            "source": ["leichte_sprache"] * len(ls_dataset),
        }
    )
    news = Dataset.from_dict(
        {
            "text": news_dataset["content"],
            "label": [SG_LABEL] * len(news_dataset),
            "source": ["news"] * len(news_dataset),
        }
    )
    wiki = Dataset.from_dict(
        {
            "text": wiki_dataset["content"],
            "label": [SG_LABEL] * len(wiki_dataset),
            "source": ["wiki"] * len(wiki_dataset),
        }
    )
    web = Dataset.from_dict(
        {
            "text": web_texts,
            "label": [SG_LABEL] * len(web_texts),
            "source": ["web"] * len(web_texts),
        }
    )

    labels = ClassLabel(num_classes=2, names=[SG_LABEL, LS_LABEL])
    features = Features(
        {"text": Value("string"), "label": labels, "source": Value("string")}
    )
    dataset = concatenate_datasets([ls, news, wiki, web]).cast(features)

    push_to_hf_hub(dataset, os.getenv("HF_CLASSIFICATION_DATASET_NAME"))
    return


if __name__ == "__main__":
    create_classification_dataset()
