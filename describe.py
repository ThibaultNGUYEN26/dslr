#!/usr/bin/env python3
import logging
import argparse

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


def parse_arguments():
    parser = argparse.ArgumentParser(description="Describe numerical CSV features.")
    parser.add_argument("dataset", help="CSV dataset path")
    parser.add_argument(
        "--bonus",
        action="store_true",
        help="include bonus statistics: missing, variance, range, and IQR",
    )
    return parser.parse_args()


def main():
    configure_logger()

    args = parse_arguments()

    try:
        print(describe(args.dataset, bonus=args.bonus))
    except DescribeError as error:
        LOGGER.error("%s", error)
        return 1
    except OSError as error:
        LOGGER.error("%s", error)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
