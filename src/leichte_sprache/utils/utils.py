from datetime import datetime
from itertools import islice
import locale
import logging
from typing import Generator


def get_logger() -> logging.Logger:
    """Get a logger with a handler for console logging. Log level is debug.
    All messages are formatted with a timestamp and the log level.

    :return: logger object
    """
    logger = logging.getLogger("ls_logger")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(formatter)

        # Add the console handler to the logger
        logger.addHandler(console_handler)
    return logger


def parse_german_date(date_string: str, format_string: str) -> datetime:
    """Parse a date with a spelled-out German month name.

    :param date_string: string containing the date, i. e. "30. März 2018, 15:30 Uhr"
    :param format_string: format string, i. e. "%d. %B %Y, %H:%M Uhr"
    :return: corresponding datetime object
    """
    locale.setlocale(locale.LC_ALL, "de_DE.utf8")
    date = datetime.strptime(date_string, format_string)
    return date


def create_batches(iterable, batch_size: int = 100) -> Generator:
    """A helper function to break an iterable into batches of size batch_size."""
    it = iter(iterable)
    batch = tuple(islice(it, batch_size))
    while batch:
        yield batch
        batch = tuple(islice(it, batch_size))
