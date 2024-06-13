import pytest

from leichte_sprache.evaluation.score import split_text_naive, calc_mean_word_length


@pytest.mark.parametrize(
    "text, sep, separate_hyphens, expected",
    (
        ["foo bar", None, True, 2],  # default case
        ["foo - bar", None, True, 2],  # hyphen with whitespace
        ["This sent is hyphen-ated", None, True, 5],  # test hyphen separation
        ["This sent is hyphen-ated", None, False, 4],  # test hyphen separation
        ["Split at: a different char", ":", True, 2],
        ["Combine: diff-erent parameters", ":", True, 3],
    ),
)
def test_count_words_naive(text: str, sep: str, separate_hyphens: bool, expected: int):
    words = split_text_naive(text=text, sep=sep, separate_hyphens=separate_hyphens)
    assert len(words) == expected
    return


@pytest.mark.parametrize(
    "text, sep, separate_hyphens, expected",
    (
        ["foo bar", None, True, 3.0],  # default case
        ["foo - bar", None, True, 3.0],  # hyphen with whitespace
        ["This sent is hyphen-ated", None, True, 4.0],  # test hyphen separation
        ["This sent is hyphen-ated", None, False, 5.25],  # test hyphen separation
        ["Split at: a different char", ":", True, 12.5],
        ["Combine: diff-erent parameters", ":", True, 9.33],
    ),
)
def test_calc_mean_word_length(
    text: str, sep: str, separate_hyphens: bool, expected: float
):
    words = split_text_naive(text=text, sep=sep, separate_hyphens=separate_hyphens)
    avg_len = calc_mean_word_length(words)
    assert round(avg_len, 2) == expected
    return
