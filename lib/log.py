import logging


def log(msg):
    """
    This function records log messages into a file named 'mylog.log'

    Argument:
    msg: The message to be logged.

    Returns:
    none
    """

    logging.basicConfig(
        filename="log/info.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logging.info(msg)
