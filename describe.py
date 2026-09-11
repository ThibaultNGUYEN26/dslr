#!/usr/bin/env python3
import logging
import sys

from src.data_analysis.describe import DescribeError, describe


LOGGER = logging.getLogger("dslr.describe")


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
    }
    RESET = "\033[0m"

    def format(self, record):
        original_levelname = record.levelname
        color = self.COLORS.get(record.levelno, "")
        if color:
            record.levelname = f"{color}{record.levelname}{self.RESET}"
        message = super().format(record)
        record.levelname = original_levelname
        return message


def configure_logger():
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter("%(levelname)s: %(message)s"))

    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.propagate = False


def main():
    configure_logger()

    if len(sys.argv) != 2:
        LOGGER.error("usage: %s <dataset.csv>", sys.argv[0])
        return 1

    try:
        print(describe(sys.argv[1]))
    except DescribeError as error:
        LOGGER.error("%s", error)
        return 1
    except OSError as error:
        LOGGER.error("%s", error)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
