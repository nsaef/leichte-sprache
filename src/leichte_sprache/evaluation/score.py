import evaluate
from lingua import Language, LanguageDetectorBuilder


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


def count_words_naive(text: str, sep: str = None, separate_hyphens: bool = True) -> int:
    """Naive word count estimate. Splits the text at the given separator
    and returns its length in words. Defaults to splitting at whitespace.

    :param text: input string
    :param sep: separator. Default: None
    :param separate_hyphens: replace hypthens with the separator token to count hyphenated words as multiple words. Default:True
    :return: number of words in the text (rough estimate)
    """
    if separate_hyphens:
        replacement = " " if sep is None else sep
        text = text.replace("-", replacement)
    split = text.split(sep)
    return len(split)


def score_classification_set():
    # get the dataset
    # run readibility, lexical diversity, avg sentence length, avg word length, avg text length?
    # -> sentence length and word length SHOULD be comprised in readibility
    # -> text length would be biased a lot by the very short dictionary entries
    pass


if __name__ == "__main__":
    score_classification_set()
