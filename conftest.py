import pytest

from leichte_sprache.utils.db_utils import get_connector


@pytest.fixture(scope="function")
def conn():
    connector = get_connector("tests/test.db")
    return connector
