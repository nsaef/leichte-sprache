from argparse import ArgumentParser
from datetime import datetime
from hashlib import md5
import io
import locale
import re
from time import sleep
import unicodedata

import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.common.exceptions import StaleElementReferenceException
from tqdm import tqdm

from leichte_sprache.constants import (
    CRAWLER_TABLE,
    SRC_COLUMN,
    ID_COLUMN,
    URL_COLUMN,
    CRAWL_TS_COLUMN,
    TEXT_COLUMN,
    TITLE_COLUMN,
    RELEASE_COLUMN,
    FULL_TEXT_COLUMN,
    DLF_DICT,
    DLF_NEWS,
    NDR,
    MDR_DICT,
    MDR_NEWS,
    HURRAKI,
    PARLAMENT,
)
from leichte_sprache.dataset.parse_pdfs import extract_pdf_das_parlament
from leichte_sprache.utils.db_utils import (
    insert_rows,
)
from leichte_sprache.utils.utils import parse_german_date


def make_soup(
    url: str,
    total_retries: int = 5,
    backoff_factor: float = 1.0,
    max_long_retries: int = 5,
    long_retry_delay: int = 30,
) -> BeautifulSoup:
    """Helper function to create a BS4 object. Includes two different retry mechanisms:
    - URLLib retries: python URLLib standard implementation for short-delay retries. Can be configured via `total_retries` and `backoff_factor`.
    - long delay retries: if the short URLLib retries fail, try waiting a longwer amount of time before retrying. Can be configured via `max_long_retries` and `long_retry_delay`.

    :param url: page URL
    :param total_retries: max retries using the URLLib retry functionality. Default: 5
    :param backoff_factor: factor for exponentiall backoff for URLLib retries in seconds. Default: 1
    :param max_long_retries: Maximum allowed number of retries with a long delay. Default: 5
    :param long_retry_delay: Long delay before retrying, to avoid spamming a website with requests. Default: 30 seconds.
    :raises requests.exceptions.HTTPError: if the website can't be retrieved, raise an HttpError
    :return: bs4 soup object
    """
    s = requests.Session()
    retries = Retry(
        total=total_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
    )
    s.mount("http://", HTTPAdapter(max_retries=retries))

    retries = 1
    success = False
    while not success and retries < max_long_retries:
        response = requests.get(url)
        if response.status_code == 200:
            success = True
        else:
            wait = retries * long_retry_delay
            sleep(wait)
            retries += 1

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
    """Create a result dict in the correct format to insert it into the `crawled_texts` table.
    The texts are processed by running unicode normalizazion and by stripping them.
    An additional key-value-pair `full_text` is created by concatenating the title and the
    text. An ID is created by hashing `full_text`.

    Output dict format:
    {
        "source": source,
        "text": text,
        "url": url,
        "crawl_timestamp": crawl_date,
        "title": title,
        "release_date": orig_date,
        "full_text": f"{title}\n{text}",
        "id": md5(full_text.encode("utf-8")).hexdigest(),
    }

    :param text: text of the page content (usually: in Leichte Sprache)
    :param source: name of the source website
    :param url: URL from which the content was crawled
    :param crawl_date: timestamp (datetime) when the content was crawled
    :param title: optional text title (f.i. article headline). Default: None
    :param orig_date: optional original release date of the text. Defaults: None
    :return: dict ready to be inserted into the `crawled_texts` table
    """
    text = unicodedata.normalize("NFKD", text.strip())
    full_text = f"{title}\n{text}".strip()
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
    """Given the content of a request to a DLF wörterbuch page for a letter
    containing multiple dictionary entries, extract the text, title,
    and release date of each entry on the page. Create a dict ready to be
    inserted into the `crawled_texts` DB  for each entry.

    :param content: content of a GET request to the URL
    :param url: URL of the page for the letter
    :return: result dicts for all entries on the page
    """
    results = []
    soup = BeautifulSoup(content, "html.parser")
    entries = soup.select("div.b-teaser-word")

    for entry in entries:
        paragraphs = [p.strip() for p in entry.text.strip().split("\n")]
        title = paragraphs[0]
        text = "\n".join(paragraphs[1:])
        res = make_result(
            text=text,
            source="dlf",
            url=url,
            crawl_date=datetime.now(),
            title=title,
        )
        results.append(res)
    return results


def get_dlf_woerterbuch():
    """
    Crawl the DLF dictionary page. There's a separate URL for each letter.
    After crawling the complete dictionary, insert all data into the DB table
    `crawled_texts`.
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


def dlf_news_load_all(landing_page_url: str, sleep_delay: int = 5) -> list[str]:
    """Crawl the DLF news articles in Leichte Sprache. On the landing page,
    automatically click the "load more" button until it disappears or at most
    1000 times. Wait a few seconds after each click to give the page time to load.
    Once all articles are loaded, collect all links to articles.

    :param landing_page_url: URL of the landing page for DLF news in Leichte Sprache
    :param sleep_delay: number of seconds to wait after loading more articles. Default: 5.
    :return: list of all DLF Leichte Sprache news article URLs
    """
    driver = webdriver.Chrome()
    driver.get(landing_page_url)
    btn = driver.find_element(By.CSS_SELECTOR, "span.content-loader-btn-more-text")
    i = 0

    while i < 1000:
        try:
            btn.click()
            sleep(sleep_delay)
            i += 1
        except StaleElementReferenceException:
            i = 9999

    article_links = driver.find_elements(By.CSS_SELECTOR, "article a")
    url_list = [link.get_attribute("href") for link in article_links]
    return url_list


def parse_dlf_article(url: str) -> dict:
    """Parse a DLF news article in Leichte Sprache. Create a BeautifulSoup
    instance from its website, extract the article text, title, teaser and
    release date. Concatenate the teaser and the text body to get as much
    text as possible. Create a dictionary from this data which can be inserted
    into the `crawled_texts` DB table and return it.

    :param url: article URL
    :return: result dictionary
    """
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
    """Crawl the news articles in Leichte Sprache in the DLF website.
    For each of their news categories, retrieve all article URLs from
    the landing page. After collecting all URLs, parse the actual page
    contents and retrieve the article text, title and release dates.
    Store the results in the DB table `crawled_texts`.
    """
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
    """Crawl the DLF content in Leichte Sprache.

    :param crawl_dict: Crawl the dictionary of Leichte Sprache terms
    :param crawl_news: Crawl the news articles
    """
    if crawl_dict:
        get_dlf_woerterbuch()
    if crawl_news:
        get_dlf_news()


def get_links_per_page(page_url: str, selector: str, url_prefix: str = "") -> list[str]:
    """Get all links from an NDR Leichte Sprache page.

    :param url: page URL
    :param selector: CSS selector for the link elements
    :param url_prefix: Optional base URL (f.i. "http://www.ndr.de"). All URLs are prefixed with this. Defaults to an empty string.
    :return: list containing all links on the page found by the given selector
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


def ndr_get_article(url: str) -> dict:
    """Parse an NDR news article in Leichte Sprache.
    Create a BeautifulSoup instance and extract the release date, title
    and text of the article. Create and return a result dictionary ready
    to be inserted into the DB table `crawled_texts`.

    :param url: article URL
    :return: dict with article data
    """
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


def ndr_get_more_news(url: str) -> list[dict]:
    """
    Parse an NDR Leichte Sprache "More news from date X" page. These pages
    contain multiple brief articles from a certain date. Separate them
    from each other and parse them all as distinct articles. Extract the
    headline (not always present), text and release date of each article
    and create a result dict per article.

    :param url: page URL
    :return: list of result dictionaries, one per article on the page
    """
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


def parse_ndr_articles(url: str) -> list[dict]:
    """Parse NDR news articles in Leichte Sprache. Call the
    appropriate parser function depending on whether it's a
    single article or a collection of articles on a page containing
    "more news from date X". Return the parsing result.

    :param url: page URL
    :return: list of result dictionaries
    """
    if "Mehr-Nachrichten-vom" in url:
        return ndr_get_more_news(url)
    else:
        return [ndr_get_article(url)]


def crawl_ndr():
    """Crawl the NDR content in Leichte Sprache by collecting all links
    to news articles and parsing their contents. Insert them into the DB
    table `crawled_texts`.
    """
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
        articles.extend(res)

    insert_rows(table_name=CRAWLER_TABLE, rows=articles)
    return


def parse_mdr_article(page_url: str) -> dict:
    """Parse an MDR page in Leichte Sprache (news article or dictionary entry).
    Extract the title, text and release date of the page. The locale needs to be set
    to German, as they use German month names in the release dates.

    :param page_url: URL of the MDR article or dictionary entry
    :return: result dictionary containign the crawled data
    """
    locale.setlocale(locale.LC_ALL, "de_DE.utf8")
    soup = make_soup(page_url)
    title = soup.select_one("h1").text.strip()
    release_date_el = soup.select_one("p.webtime")
    if release_date_el:
        release_date = parse_german_date(
            date_string=release_date_el.text.strip(),
            format_string="%d. %B %Y, \n%H:%M Uhr",
        )
    else:
        release_date = None
    paragraphs = soup.select("div.paragraph p")
    text = "\n".join(p.get_text("\n").strip() for p in paragraphs)
    result = make_result(
        text=text,
        source="mdr",
        url=page_url,
        crawl_date=datetime.now(),
        title=title,
        orig_date=release_date,
    )
    return result


def get_mdr_worterbuch():
    """Collect all links on the MDR dictionary page, then extract and
    parse the article contents. Create a result dict for each entry
    and store all results in the DB table `crawled_texts`.
    """
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
    """Crawl the MDR news articles in Leichte Sprache. They are organized
    by land - collect the links to all articles for each land, then extract
    the article contents, create result dicts and store them in the DB table
    `crawled_texts`.
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
    """Crawl the MDR texts in Leichte Sprache.

    :param crawl_dict: Crawl the MDR dictionary of terms in Leichte Sprache
    :param crawl_news: Crawl the MDR news articles in Leichte Sprache
    """
    if crawl_dict:
        get_mdr_worterbuch()
    if crawl_news:
        get_mdr_articles()
    return


def parse_hurraki_article(url: str) -> dict:
    """Crawl a Hurraki wiki page. Ignore the section that refers similar works
    and remove the wiki-elements such as edit links on subheadings.

    :param url: article URL
    :return: result dictionary for the page
    """
    soup = make_soup(url)

    title = soup.select_one("h1").text

    synonyms_text = "Gleiche Wörter[Bearbeiten | Quelltext bearbeiten]"
    content = soup.select_one("div#mw-content-text div.mw-parser-output")
    text = ""
    for child in content.children:
        if child.name not in ["p", "h2"]:
            continue
        if child.name == "h2" and child.text == synonyms_text:
            continue
        previous_subheading = child.find_previous("h2")
        if previous_subheading and previous_subheading.text == synonyms_text:
            continue
        par_text = child.text.replace("[Bearbeiten | Quelltext bearbeiten]", "")
        cleaned = re.sub(r"\n{3,}", "\\n\\n", par_text)
        text += cleaned
    result = make_result(
        text=text, source="hurraki", url=url, crawl_date=datetime.now(), title=title
    )
    return result


def crawl_hurraki():
    """Crawl the Hurraki wiki in Leichte Sprache."""
    base_url = "https://hurraki.de/wiki/Hurraki:Artikel_von_A_bis_Z"
    selector = "#mw-content-text > div.mw-parser-output > table:nth-child(3) a"
    links = get_links_per_page(
        base_url, selector=selector, url_prefix="https://hurraki.de"
    )
    articles = []

    for link in tqdm(links, desc="Parsing Hurraki articles"):
        if "index.php" in link:
            continue
        res = parse_hurraki_article(link)
        articles.append(res)

    insert_rows(table_name=CRAWLER_TABLE, rows=articles)
    return


def get_valid_parlament_editions(links: list[str]) -> list[str]:
    """#todo _summary_

    :param links: _description_
    :return: _description_
    """
    valid_links = []

    # remove old editions without Leichte Sprache
    rx = r"de\/epaper\/(\d{4})\/([\d_]+)\/"
    for link in links:
        match = re.search(rx, link)
        year, edition = int(match.group(1)), match.group(2)
        if year < 2014:
            continue
        if year == 2014 and int(edition.split("_")[0]) < 27:
            continue
        valid_links.append(link)
    return valid_links


def parse_parlament_edition(url: str):
    """#todo_summary_

    :param url: _description_
    :return: _description_
    """
    all_results = []
    r = requests.get(url)
    pdf_io = io.BytesIO(r.content)
    data = extract_pdf_das_parlament(pdf_io)
    for article in data:
        res = make_result(
            source="das_parlament", url=url, crawl_date=datetime.now(), **article
        )
        all_results.append(res)
    return all_results


def crawl_das_parlament():
    """#todo _summary_"""
    all_links = get_links_per_page(
        "https://www.das-parlament.de/e-paper",
        selector="a.epaper__link[title~='PDF']",
        url_prefix="https://www.das-parlament.de",
    )
    links = get_valid_parlament_editions(all_links)
    results = []

    for link in tqdm(links):
        res = parse_parlament_edition(link)
        results.extend(res)
    results = [res for res in results if res.get("text")]
    insert_rows(table_name=CRAWLER_TABLE, rows=results)
    return


def run_crawler(
    dlf_dict: bool,
    dlf_news: bool,
    ndr: bool,
    mdr_dict: bool,
    mdr_news: bool,
    hurraki: bool,
    parlament: bool,
):
    """Crawl content in Leichte Sprache.

    :param dlf_dict: Crawl the DLF dictionary
    :param dlf_news: Crawl the DLF news
    :param ndr: Crawl the NDR news
    :param mdr_dict: Crawl the MDR dictionary
    :param mdr_news: Crawl the MDR news
    :param hurraki: Crawl the Hurraki wiki in Leichte Sprache
    :param parlament: Crawl "Das Parlament" (newspaper of the German Bundestag with four pages Leichte Sprache)
    """
    crawl_dlf(crawl_dict=dlf_dict, crawl_news=dlf_news)
    crawl_mdr(crawl_dict=mdr_dict, crawl_news=mdr_news)
    if ndr:
        crawl_ndr()
    if hurraki:
        crawl_hurraki()
    if parlament:
        crawl_das_parlament()
    return


def parse_args() -> ArgumentParser:
    """Parse the command line arguments to select the sources to crawl.

    :return: ArgumentParser with command line arguments.
    """
    parser = ArgumentParser(
        prog="Leichte Sprache Crawler", description="Crawl content in Leichte Sprache"
    )
    parser.add_argument(
        "--sources",
        choices=[DLF_NEWS, DLF_DICT, NDR, MDR_DICT, MDR_NEWS, HURRAKI, PARLAMENT],
        nargs="*",
        help="Select the sources to crawl. If this argument is not used, all sources will be crawled.",
    )
    # parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    run_crawler(
        dlf_dict=True if not args.sources or DLF_DICT in args.sources else False,
        dlf_news=True if not args.sources or DLF_NEWS in args.sources else False,
        ndr=True if not args.sources or NDR in args.sources else False,
        mdr_dict=True if not args.sources or MDR_DICT in args.sources else False,
        mdr_news=True if not args.sources or MDR_NEWS in args.sources else False,
        hurraki=True if not args.sources or HURRAKI in args.sources else False,
        parlament=True if not args.sources or PARLAMENT in args.sources else False,
    )
