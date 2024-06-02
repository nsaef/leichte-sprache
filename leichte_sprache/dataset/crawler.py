from datetime import datetime
from hashlib import md5
import locale
from time import sleep
import unicodedata

import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException
from tqdm import tqdm

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
    insert_rows,
)


def make_soup(url: str) -> BeautifulSoup:
    """Helper function to create a BS4 object.

    :param url: page URL
    :raises requests.exceptions.HTTPError: if the website can't be retrieved, raise an HttpError
    :return: bs4 soup object
    """
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        return soup
    else:
        raise requests.exceptions.HTTPError(
            f"Request failed. Status code: {response.status_code}"
        )


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
    text = unicodedata.normalize("NFKD", text.strip())
    full_text = f"{title}\n{text}"
    result = {
        SRC_COLUMN: source,
        TEXT_COLUMN: text,
        URL_COLUMN: url,
        CRAWL_TS_COLUMN: crawl_date,
        TITLE_COLUMN: title,
        RELEASE_COLUMN: orig_date,
        FULL_TEXT_COLUMN: full_text,
        ID_COLUMN: md5(full_text.encode("utf-8")).hexdigest(),
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
    i = 0

    # click the "load more" button until it disappears or max 1000 times
    while i < 1000:
        try:
            btn.click()
            sleep(5)
            i += 1
        except StaleElementReferenceException:
            i = 9999

    # retrieve all article URLs from the website
    article_links = driver.find_elements(By.CSS_SELECTOR, "article a")
    url_list = [link.get_attribute("href") for link in article_links]
    return url_list


def parse_dlf_article(url):
    soup = make_soup(url)
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


def get_links_per_page(page_url: str, selector: str, url_prefix: str = "") -> list[str]:
    """Get all links from an NDR Leichte Sprache archive page.
    #todo: update docstring for generalized function, add parameters

    :param url: archive page URL
    :return: list containing all links on the page
    """
    soup = make_soup(page_url)
    link_elements = soup.select(selector)
    links = [url_prefix + el.get("href") for el in link_elements]
    return links


def ndr_get_release_date(soup) -> datetime:
    """Parse the article release date (more precisely: the last update) from NDR articles
    into a datetime object.

    :param soup: BeautifulSoup object of an NDR website
    :return: datetime object with the article release/update date
    """
    release_date_str = soup.select_one("header div.lastchanged").text.strip("\nStand: ")
    release_date = datetime.strptime(release_date_str, "%d.%m.%Y %H:%M Uhr")
    return release_date


def ndr_get_paragraphs(soup, article_end_str: str) -> list[str]:
    """Get the paragraphs containing the article text. Filter empty paragraphs and use
    the provided `article_end_str` to find the end of the article text.

    :param soup: BeautifulSoup object of the article page
    :param article_end_str: text that identifies the first paragraph after the article text
    :return: list of paragraphs containing the article text
    """
    paragraphs = soup.select("article p")
    par_texts = [p.text.strip() for p in paragraphs if p.text.strip()]

    last_index = None
    for i, t in enumerate(par_texts):
        if t.startswith(article_end_str):
            last_index = i
        elif t.startswith("Dann klicken Sie hier"):
            last_index = i + 1
    filtered_paragraphs = par_texts[:last_index]
    return filtered_paragraphs


def ndr_get_article(url: str):
    # todo: docstring
    soup = make_soup(url)
    release_date = ndr_get_release_date(soup)
    title = soup.select_one("header h1").text.strip()
    paragraphs = ndr_get_paragraphs(soup, article_end_str="Diese Nachricht ist vom ")
    text = "\n".join(paragraphs)

    res = make_result(
        text=text,
        source="ndr",
        url=url,
        crawl_date=datetime.now(),
        title=title,
        orig_date=release_date,
    )
    return res


def ndr_get_more_news(url: str):
    soup = make_soup(url)

    # get content from website
    release_date = ndr_get_release_date(soup)
    headline_elements = soup.select("article div h2")
    headlines = [h.text.strip() for h in headline_elements if h.text.strip()]
    paragraphs = ndr_get_paragraphs(soup, article_end_str="Diese Nachrichten sind vom ")

    # parse into separate articles
    text = ""
    results = []
    i = 0

    for par in paragraphs:
        if len(par) == par.count("-"):
            # end of article reached - create a result
            try:
                headline = headlines[i]
            except IndexError:
                headline = None
            res = make_result(
                text=text,
                source="ndr",
                url=url,
                crawl_date=datetime.now(),
                title=headline,
                orig_date=release_date,
            )
            results.append(res)
            text = ""
            i += 1
        else:
            text += "\n" + par
    return results


def parse_ndr_articles(url: str):
    if "Mehr-Nachrichten-vom" in url:
        return ndr_get_more_news(url)
    else:
        return ndr_get_article(url)


def crawl_ndr():
    # todo docstring
    urls = []
    articles = []
    page_nrs = range(1, 101)

    for page_nr in tqdm(page_nrs, desc="Collecting NDR links"):
        page_url = f"https://www.ndr.de/fernsehen/barrierefreie_angebote/leichte_sprache/leichtesprachearchiv110_page-{page_nr}.html"
        links = get_links_per_page(
            page_url=page_url, selector="article h2 a", url_prefix="https://www.ndr.de"
        )
        urls.extend(links)

    for url in tqdm(urls, desc="Crawling NDR articles"):
        if "Jahresrueckblick-in-Leichter-Sprache" in url:
            continue
        res = parse_ndr_articles(url)
        if isinstance(res, list):
            articles.extend(res)
        else:
            articles.append(res)

    insert_rows(table_name=CRAWLER_TABLE, rows=articles)
    return


def parse_mdr_article(article_url: str) -> dict:
    """_summary_
    #todo docstring
    :param article_url: _description_
    :return: _description_
    """
    locale.setlocale(locale.LC_ALL, "de_DE.utf8")
    soup = make_soup(article_url)
    title = soup.select_one("h1").text.strip()
    release_date_el = soup.select_one("p.webtime")
    if release_date_el:
        # release_date_str = soup.select_one("p.webtime").text.strip()
        release_date = datetime.strptime(
            release_date_el.text.strip(), "%d. %B %Y, \n%H:%M Uhr"
        )
    else:
        release_date = None
    paragraphs = soup.select("div.paragraph p")
    text = "\n".join(p.get_text("\n").strip() for p in paragraphs)
    result = make_result(
        text=text,
        source="mdr",
        url=article_url,
        crawl_date=datetime.now(),
        title=title,
        orig_date=release_date,
    )
    return result


def get_mdr_worterbuch():
    base_url = "https://www.mdr.de/nachrichten-leicht/woerterbuch/index.html"
    links = get_links_per_page(
        page_url=base_url,
        selector="div.multiGroupNavi a",
        url_prefix="https://www.mdr.de",
    )
    article_urls = []
    results = []

    for link in tqdm(links, desc="Collecting MDR Wörterbuch links"):
        crawled_article_urls = get_links_per_page(
            page_url=link, selector="a.linkAll", url_prefix="https://www.mdr.de"
        )
        article_urls.extend(crawled_article_urls)

    for article_url in tqdm(article_urls, desc="Crawling MDR Wörterbuch"):
        if (
            article_url
            == "https://www.mdr.de/nachrichten-leicht/woerterbuch/glossar-leichte-sprache-100.html"
        ):
            continue
        res = parse_mdr_article(article_url)
        results.append(res)

    insert_rows(table_name=CRAWLER_TABLE, rows=results)
    return


def get_mdr_articles():
    """
    # todo: docstring
    """
    laender = ["sachsen", "sachsen-anhalt", "thueringen"]
    article_urls = []
    results = []

    for land in tqdm(laender, desc="Collecting links on subpages"):
        base_url = f"https://www.mdr.de/nachrichten-leicht/rueckblick/leichte-sprache-rueckblick-buendelgruppe-{land}-100.html"
        links_per_page = get_links_per_page(
            page_url=base_url, selector="a.linkAll", url_prefix="https://www.mdr.de"
        )
        article_urls.extend(links_per_page)

    for link in tqdm(links_per_page, desc="Crawling MDR articles"):
        if "rueckblick-buendelgruppe" in link or link.endswith(
            "nachrichten-in-leichter-sprache-114.html"
        ):
            continue
        res = parse_mdr_article(link)
        results.append(res)

    insert_rows(table_name=CRAWLER_TABLE, rows=results)
    return


def crawl_mdr(crawl_dict: bool, crawl_news: bool):
    if crawl_dict:
        get_mdr_worterbuch()
    if crawl_news:
        get_mdr_articles()


def run_crawler(dlf_dict: bool, dlf_news: bool, ndr: bool, mdr: bool):

    crawl_dlf(crawl_dict=dlf_dict, crawl_news=dlf_news)

    if ndr:
        crawl_ndr()

    if mdr:
        crawl_mdr(crawl_dict=True, crawl_news=True)
    return


if __name__ == "__main__":
    # todo add argparse

    run_crawler(
        dlf_dict=False,
        dlf_news=False,
        ndr=False,
        mdr=True,
    )
