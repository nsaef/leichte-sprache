import os

from py_pdf_parser.loaders import load_file
from tqdm import tqdm


def column_ordering_function(elements):
    """
    The first entry in the key is False for columm 1, and True for column 2. The second
    and third keys just give left to right, top to bottom.
    """
    return sorted(elements, key=lambda elem: (elem.x0 > 450, -elem.y0, elem.x0))


def extract_pdf_das_parlament(fpath: str):
    """
    #todo
    :param fpath: path of a "Das Parlament" pdf file
    """
    ARTICLE_TITLE = "article_title"
    ARTICLE_SUBTITLE = "article_subtitle"
    TITLE = "title"
    LOGO = "logo"
    SUBTITLE = "subtitle"
    ARTICLE_TEXT = "article_text"

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
    }

    doc = load_file(
        fpath,
        font_mapping=FONT_MAPPING,
        font_mapping_is_regex=True,
        element_ordering=column_ordering_function,
    )
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


if __name__ == "__main__":
    dirname = "/home/nasrin/workspace/leichte-sprache/data/pdf_examples"
    filenames = os.listdir(dirname)

    articles = []
    for filename in tqdm(filenames):
        fpath = f"{dirname}/{filename}"
        res = extract_pdf_das_parlament(fpath)
        articles.extend(res)

    for article in articles:
        print(article.get("title") + "\n")
        print(article.get("text") + "\n\n - - - - - \n\n")
    print("hello world")
