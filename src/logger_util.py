import logging
import os


def get_logger():
    parent_dir = os.path.abspath(".")
    file_name = "corpus_cleaner.log"
    file_path = os.path.join(parent_dir, "corpus_cleaner", "log", file_name)

    fmt = "%(asctime)s %(levelname)s %(filename)s " \
          "%(lineno)d %(process)d %(message)s"
    datefmt = "%a, %d %b %Y %H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt)

    stream_handle = logging.StreamHandler()
    stream_handle.setFormatter(formatter)
    stream_handle.setLevel(logging.WARN)

    file_handle = logging.FileHandler(filename=file_path)
    file_handle.setFormatter(formatter)
    file_handle.setLevel(logging.DEBUG)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handle)
    logger.addHandler(file_handle)
    return logger


if __name__ == "__main__":

    logging.debug("debug message...")
    logging.info("info message...")
    logging.warn("warn message...")
    logging.error("error message...")
    logging.critical("critical message...")

    path = os.getcwd()
    parent_path = os.path.abspath("..")
    print("path ==> %s" % path)
    print("parent_path ==> %s" % parent_path)
