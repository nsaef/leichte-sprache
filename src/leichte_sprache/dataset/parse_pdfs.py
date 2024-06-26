from py_pdf_parser.loaders import load

from leichte_sprache.constants import (
    ARTICLE_TEXT,
    ARTICLE_TITLE,
    ARTICLE_SUBTITLE,
    TITLE,
    LOGO,
    SUBTITLE,
)
from leichte_sprache.utils.utils import parse_german_date


def column_ordering_function(elements):
    """
    Parse a PDF with a two-column layout. The x-coordinate of the divider delimits the column.
    """
    divider_position = 450
    return sorted(
        elements, key=lambda elem: (elem.x0 > divider_position, -elem.y0, elem.x0)
    )


def extract_pdf_das_parlament(f: bytes) -> list[dict]:
    """
    Extract the content of PDFs of Das Parlament.

    The parsing depends heavily on constant layout rules. The following assumptions are made:

    - the content in Leichte Sprache is always precisely on the last four pages
    - it is always presented in a two-column layout
    - certain fonts are used continually for the same elements
    - the header row is always at approximately the same height

    With these assumptions in mind:
    - the date is extracted from the first page; the format changed some time since 2014, and only one date format is supported
    - all text elements are extracted from the last four pages in Leichte Sprache
    - the articles in Leichte Sprache are parsed using the articel headings, subheadings and texts

    :param f: Bytes object of the PDF file
    :return: list with one result dictionary per article
    """

    FONT_MAPPING = {
        r"\w{6}\+TheMixOffice-Bold,16.0": ARTICLE_TITLE,
        r"\w{6}\+TheMixOffice-Bold,14.0": ARTICLE_SUBTITLE,
        r"\w{6}\+TheMixOffice-Bold,58.2": TITLE,
        r"\w{6}\+TheMixOffice-Bold,58.2": TITLE,
        r"\w{6}\+TheMixOffice-Bold,40.9": TITLE,
        r"\w{6}\+TheMixOffice-Regular,26.1": TITLE,
        r"\w{6}\+TheMixOffice-Bold,51.1": LOGO,
        r"\w{6}\+TheMixOffice-Regular,26.9": SUBTITLE,
        r"\w{6}\+TheMixOffice-Regular,14.0": ARTICLE_TEXT,
        r"\w{6}\+TheMixOffice-Regular,16.0": ARTICLE_TITLE,
        r"\w{6}\+FrutigerLTPro-Bold,10.0": "date",
    }

    doc = load(
        f,
        font_mapping=FONT_MAPPING,
        font_mapping_is_regex=True,
        element_ordering=column_ordering_function,
    )
    try:
        date_str = doc.elements.filter_by_font("date")[0].text().split(",")[1].strip()
        date = parse_german_date(date_string=date_str, format_string="%d. %B %Y")
    except (IndexError, ValueError) as e:
        date = None

    ls_pages = doc.pages[-4:]
    all_elements = [ele for page in ls_pages for ele in page.elements]

    all_articles = []
    article = None
    article_title = None

    for ele in all_elements:
        # skip headers
        if ele.bounding_box.y0 > 1000 and ele.bounding_box.y1 > 1000:
            continue
        text = ele.text(stripped=False)
        font = ele.font
        # get the article title - needed to parse the header line, in case the y coordinates don't match
        if font == TITLE and not article_title:
            article_title = text.replace("\n", "").strip()
        # parse the article title and use any article title as the indicator that the current article is finished
        elif font == ARTICLE_TITLE:
            if article:
                all_articles.append(article)
            article = {
                "title": text,
                "text": "",
                "orig_date": date,
            }
        elif font == ARTICLE_TEXT:
            # reached the end of the content
            if text.replace("\n", "").startswith(
                "Weitere Informationen in Leichter Sprache gibt es unter:"
            ):
                all_articles.append(article)
                break
            # skip intro lines
            if text.strip() in [
                "Informationen in Leichter Sprache",
                "Beilage für:",
            ] or text.startswith("Ausgabe Nr. "):
                continue
            # skip header lien (if it's at unexpected coordinates)
            elif article_title and (
                text.startswith(article_title)
                and "•" in text
                or text.replace(" • ", " ").strip()
                == article_title.replace("\n", " ").strip()
            ):
                continue
            # add the actual article text
            elif article:
                article["text"] += text + " "
        # no special tratment for article subheadings - just add them to the text with a preceding newline
        elif font == ARTICLE_SUBTITLE and article:
            article["text"] += "\n" + text
    # early articles use lots of tabstops for formatting - replace them with regular whitespace
    for article in all_articles:
        article["text"] = article["text"].replace("\t", " ")
    return all_articles
