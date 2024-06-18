from datetime import datetime
import locale
import logging


def get_logger() -> logging.Logger:
    """Get a logger with a handler for console logging. Log level is debug.
    All messages are formatted with a timestamp and the log level.

    :return: logger object
    """
    logger = logging.getLogger("ls_logger")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

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
