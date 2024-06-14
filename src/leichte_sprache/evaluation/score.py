import os
from statistics import mean

from datasets import load_dataset
import evaluate
from lingua import Language, LanguageDetectorBuilder
import spacy


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


def split_text_naive(
    text: str, sep: str = None, separate_hyphens: bool = True
) -> list[str]:
    """Naive word splitting. Splits the text at the given separator and returns a list of words.
    Defaults to splitting at whitespace. By default, also replaces hyphens (which are commonly
    inserted into long words in Leichte Sprache) with the separator to account for the added
    readibility generated by this spelling.

    :param text: input string
    :param sep: separator. Default: None
    :param separate_hyphens: replace hyphens with the separator token to count hyphenated words as multiple words. Default:True
    :return: list of words
    """
    if separate_hyphens:
        replacement = " " if sep is None else sep
        text = text.replace("-", replacement)
    split = text.split(sep)
    return split


def calc_mean_word_length(split_text: list[str]) -> float:
    """Calculate the mean word length of a split (tokenized) text.
    Split the text using `score.split_text_naive` with your preferred
    parameters before using this function.

    :param split_text: text split into words
    :return: mean word length across the whole text
    """
    result = mean([len(w) for w in split_text])
    return result


def get_spacy_doc(text: str) -> spacy.Language:
    """Create a spacy document from a text.

    :param text: Input text
    :return: spacy doc
    """
    nlp = spacy.load("de_core_news_sm")
    doc = nlp(text)
    return doc


def calc_mean_sents_per_paragraph(paragraphs: list[list[str]]) -> float:
    """Take a list of paragraphs, each containing a list of the sentences
    in the paragraph. Calculate the mean number of sentences inside the paragraphs.

    :param paragraphs: list if sentences in each paragraph. Example: [["this is the first paragraph", "it has two sentences"], ["this is paragraph nr. 2"]]
    :return: mean paragraph length in sentences
    """
    n_sents = [len(p) for p in paragraphs]
    mean_sents = mean(n_sents)
    return mean_sents


def split_paragraphs(sents: list) -> list[list[str]]:
    """Split a list of spacy sentences into a list of sentences per paragraph.

    :param sents: list of doc.sents
    :return: a list containing, for each paragraph, a list of the sentences in the paragraph
    """
    paragraphs = []

    current_par = []
    for sent in sents:
        current_par.append(sent)
        if "\n" in sent.text:
            paragraphs.append(current_par)
            current_par = []
    return paragraphs


def analyse_text_statistics(text: str) -> dict:
    """Run some quick statistics on the current text. Currently implemented are:

    - total number of sentences in the text
    - mean word length
    - mean paragraph length (in sentences)

    :param text: input text
    :return: dict with the metric names as keys and the results as values
    """

    # preprocess text
    naive_split = split_text_naive(text)
    doc = get_spacy_doc(text)
    sents = [sent for sent in doc.sents]
    paragraphs = split_paragraphs(sents)

    # calculate statistics
    mean_word_length = calc_mean_word_length(naive_split)
    total_sents = len(sents)
    mean_sents_per_paragraph = calc_mean_sents_per_paragraph(paragraphs=paragraphs)

    res = {
        "total_sent_number": total_sents,
        "mean_word_length": mean_word_length,
        "mean_sents_per_paragraph": mean_sents_per_paragraph,
    }
    return res


def score_classification_set():
    # get the dataset
    dataset = load_dataset(os.environ("HF_CLASSIFICATION_DATASET_NAME"), split="train")
    # run readibility, lexical diversity, avg sentence length, avg word length, avg text length?
    # -> sentence length and word length SHOULD be comprised in readibility
    # -> text length would be biased a lot by the very short dictionary entries
    pass


def run_rule_based_checks():
    # use short words, separate all long words with a hyphen
    # no abbreviations
    # use verbs rather than nouns
    # use active rather than passive
    # avoid genitive forms
    # avoid conjunctive forms
    # avoid negations
    # use arabic numbers only
    # avoid precise dates which are far in the past
    # avoid precise numbers (floating point, percentages)
    # prefer numbers (1) over number-words (one)
    # avoid special characters like " % ... ; & () $
    pass


if __name__ == "__main__":
    # score_classification_set()
    text = """Elf Menschen sind bei einem Brand in einem Hochhaus in Mainz verletzt worden, zwei von ihnen schwer. Sie erlitten Rauchgasvergiftungen, wie ein Sprecher der Feuerwehr am Freitagmorgen mitteilte. Die beiden Schwerverletzen wurden ins Krankenhaus gebracht, die anderen neun vor Ort behandelt. Die übrigen rund 120 Bewohner des Gebäudes mit 18 Etagen konnten sich teils mithilfe der Einsatzkräfte ins Freie in Sicherheit bringen. Brandursache und Schadenshöhe waren zunächst unklar.
    Die Feuerwehr war in den frühen Morgenstunden alarmiert worden, die Anrufer berichteten von Hilferufen. Als die Einsatzkräfte an dem Haus in Oberstadt eintrafen, stand die Pizzeria im Erdgeschoss in Flammen, das gesamte Erdgeschoss war verqualmt.
    Ein Übergreifen der Flammen und des Rauches auf andere Gebäudeteile konnte verhindert werden, sodass alle Wohnungen bewohnbar blieben, hieß es weiter. Die ersten Bewohner konnten morgens in ihre Wohnungen zurückkehren.
    """
    analyse_text_statistics(text)
