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
