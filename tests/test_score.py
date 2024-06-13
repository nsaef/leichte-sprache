import pytest

from leichte_sprache.evaluation.score import count_words_naive


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
def test_count_words_naive(text, sep, separate_hyphens, expected):
    words = count_words_naive(text=text, sep=sep, separate_hyphens=separate_hyphens)
    assert words == expected
    return
