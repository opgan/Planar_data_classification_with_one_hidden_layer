import logging


def log_to_file(msg):
    """
    This function records log messages into a file named 'mylog.log'

    Argument:
    msg: The message to be logged.

    Returns:
    none
    """

    logging.basicConfig(
        filename="mylog.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(msg)
