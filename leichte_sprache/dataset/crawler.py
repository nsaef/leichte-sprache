from datetime import datetime
from time import sleep

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException
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


def dlf_news_load_all(landing_page_url):
    # todo docstring
    driver = webdriver.Chrome()
    driver.get(landing_page_url)
    btn = driver.find_element(By.CSS_SELECTOR, "span.content-loader-btn-more-text")
    # article_count = 0
    # diff = 1
    i = 0

    # click the "load more" button until it disappears or max 1000 times
    while i < 1000:
        try:
            btn.click()
            sleep(5)
            all_articles = driver.find_elements(
                By.CSS_SELECTOR, "article.b-teaser-wide"
            )
            # diff = len(all_articles) - article_count
            # article_count = len(all_articles)
            i += 1
        except StaleElementReferenceException:
            i = 9999

    # retrieve all article URLs from the website
    article_links = driver.find_elements(By.CSS_SELECTOR, "article a")
    url_list = [link.get_attribute("href") for link in article_links]
    return url_list


def parse_dlf_article(url):
    res = None
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        title = soup.select("article span.headline-title")[0].text
        teaser = soup.select("p.article-header-description")[0].text.strip()
        date_str = soup.select("p.article-header-author")[0].text
        release_date = datetime.strptime(date_str, "%d.%m.%Y")
        paragraphs = soup.select("div.article-details-text")
        article_text = "\n".join([par.text for par in paragraphs])
        text = f"{teaser}\n{article_text}"
        res = make_result(
            text=text,
            source="dlf_leicht",
            url=url,
            crawl_date=datetime.now(),
            title=title,
            orig_date=release_date,
        )
    return res


def get_dlf_news():
    categories = ["nachrichten", "kultur-index", "vermischtes", "sport"]
    all_urls = []
    articles = []

    for category in categories:
        cat_url = (
            f"https://www.nachrichtenleicht.de/nachrichtenleicht-{category}-100.html"
        )
        print(f"Retrieving all URLs from {category}")
        url_list = dlf_news_load_all(cat_url)
        all_urls.extend(url_list)

    for url in tqdm(all_urls):
        res = parse_dlf_article(url)
        articles.append(res)

    insert_rows(table_name=CRAWLER_TABLE, rows=articles)
    return


def crawl_dlf(crawl_dict: bool, crawl_news: bool):
    # todo: docstring
    if crawl_dict:
        get_dlf_woerterbuch()
    if crawl_news:
        get_dlf_news()


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


def run_crawler(initial_setup: bool, dlf_dict: bool, dlf_news: bool):
    if initial_setup:
        setup_db_table()
    crawl_dlf(crawl_dict=dlf_dict, crawl_news=dlf_news)
    return


if __name__ == "__main__":
    # todo add argparse
    run_crawler(initial_setup=False, dlf_dict=False, dlf_news=True)
