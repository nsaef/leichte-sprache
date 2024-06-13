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
