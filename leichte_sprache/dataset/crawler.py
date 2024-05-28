from datetime import datetime
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from leichte_sprache.constants import CRAWLER_TABLE
from leichte_sprache.utils.db_utils import create_column_dict, create_table, insert_rows


def make_result(
    text: str,
    source: str,
    url: str,
    crawl_date: datetime,
    title: str = None,
    orig_date: datetime = None,
) -> dict:
    # todo docstring
    # todo extend
    result = {
        "source": source,
        "text": text,
        "url": url,
        "crawl_timestamp": crawl_date,
        "title": title,
        "release_date": orig_date,
    }
    return result


def parse_dlf_woerterbuch(content: bytes, url: str) -> list[dict]:
    # todo docstring
    results = []
    soup = BeautifulSoup(content, "html.parser")
    entries = soup.select("div.b-teaser-word")

    for entry in entries:
        paragraphs = [p.strip() for p in entry.text.strip().split("\n")]
        title = paragraphs[0]
        text = "\n".join(paragraphs[1:])
        res = make_result(
            text=text,
            source="dlf_leicht",
            url=url,
            crawl_date=datetime.now(),
            title=title,
        )
        results.append(res)
    return results


def get_dlf_woerterbuch():
    """
    Crawl the DLF dictionary page. There's a separate URL for each letter.
    """
    entries = []
    base_url = "https://www.nachrichtenleicht.de/woerterbuch"
    letters = [
        "1",
        "2",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]

    for letter in tqdm(letters, desc="Crawling DLF dictionary"):
        url = f"{base_url}?drsearch:letter={letter.lower()}"
        response = requests.get(url)
        if response.status_code == 200:
            entry = parse_dlf_woerterbuch(response.content, url)
            entries.extend(entry)
    insert_rows(table_name=CRAWLER_TABLE, rows=entries)
    return


def get_dlf_news():
    categories = ["nachrichten", "kultur", "vermischtes", "sport"]
    base_urls = []


def crawl_dlf():
    dict_entries = get_dlf_woerterbuch()
    # todo: crawl news articles


def setup_db_table():
    # todo: docstring
    # todo: constants for column names
    columns = [
        create_column_dict(col_name="source", col_dtype="varchar(124)", not_null=True),
        create_column_dict(col_name="text", col_dtype="text", not_null=True),
        create_column_dict(col_name="url", col_dtype="text", not_null=True),
        create_column_dict(
            col_name="crawl_timestamp", col_dtype="datetime", not_null=True
        ),
        create_column_dict(col_name="title", col_dtype="varchar(256)"),
        create_column_dict(col_name="release_date", col_dtype="datetime"),
    ]
    create_table(CRAWLER_TABLE, columns=columns)
    return


def run_crawler():
    initial_setup = True
    if initial_setup:
        setup_db_table()
    crawl_dlf()


if __name__ == "__main__":
    run_crawler()
