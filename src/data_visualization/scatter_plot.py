import argparse
import csv
import logging
import math
import os
import warnings
from pathlib import Path


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings("ignore", message="Unable to import Axes3D.*")

import matplotlib.pyplot as plt


HOUSE_COLUMN = "Hogwarts House"
IGNORED_NUMERIC_COLUMNS = {"Index"}
HOUSE_COLORS = {
    "Gryffindor": "#b31b1b",
    "Hufflepuff": "#d6a600",
    "Ravenclaw": "#1f4e8c",
    "Slytherin": "#1f7a4d",
}


class ScatterPlotError(Exception):
    pass


LOGGER = logging.getLogger("dslr.scatter_plot")


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


def parse_float(value):
    value = value.strip()
    if value == "":
        return None

    try:
        number = float(value)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None
    return number


def load_rows(csv_path):
    path = Path(csv_path)
    if not path.exists():
        raise ScatterPlotError(f"file not found: {csv_path}")
    if not path.is_file():
        raise ScatterPlotError(f"not a file: {csv_path}")

    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ScatterPlotError("empty dataset")
        if HOUSE_COLUMN not in reader.fieldnames:
            raise ScatterPlotError(f"missing column: {HOUSE_COLUMN}")
        return reader.fieldnames, list(reader)


def detect_numeric_features(fieldnames, rows):
    features = []
    for fieldname in fieldnames:
        if fieldname in IGNORED_NUMERIC_COLUMNS or fieldname == HOUSE_COLUMN:
            continue

        has_number = False
        for row in rows:
            if parse_float(row.get(fieldname, "")) is not None:
                has_number = True
                break
        if has_number:
            features.append(fieldname)

    if len(features) < 2:
        raise ScatterPlotError("dataset needs at least two numerical features")
    return features


def feature_pairs(rows, feature_x, feature_y):
    pairs = []
    for row in rows:
        x = parse_float(row.get(feature_x, ""))
        y = parse_float(row.get(feature_y, ""))
        if x is not None and y is not None:
            pairs.append((x, y, row.get(HOUSE_COLUMN, "").strip()))
    return pairs


def pearson_correlation(pairs):
    count = 0
    sum_x = 0.0
    sum_y = 0.0
    for x, y, _ in pairs:
        count += 1
        sum_x += x
        sum_y += y

    if count < 2:
        return None

    mean_x = sum_x / count
    mean_y = sum_y / count
    covariance = 0.0
    variance_x = 0.0
    variance_y = 0.0

    for x, y, _ in pairs:
        delta_x = x - mean_x
        delta_y = y - mean_y
        covariance += delta_x * delta_y
        variance_x += delta_x * delta_x
        variance_y += delta_y * delta_y

    if variance_x == 0.0 or variance_y == 0.0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)


def find_most_similar_features(csv_path):
    fieldnames, rows = load_rows(csv_path)
    features = detect_numeric_features(fieldnames, rows)

    best_feature_x = None
    best_feature_y = None
    best_correlation = None
    best_pairs = None

    for index, feature_x in enumerate(features):
        for feature_y in features[index + 1 :]:
            pairs = feature_pairs(rows, feature_x, feature_y)
            correlation = pearson_correlation(pairs)
            if correlation is None:
                continue

            if (
                best_correlation is None
                or abs(correlation) > abs(best_correlation)
            ):
                best_feature_x = feature_x
                best_feature_y = feature_y
                best_correlation = correlation
                best_pairs = pairs

    if best_correlation is None:
        raise ScatterPlotError("could not compare numerical feature pairs")

    return best_feature_x, best_feature_y, best_correlation, best_pairs


def get_feature_pair(csv_path, feature_x, feature_y):
    fieldnames, rows = load_rows(csv_path)
    features = detect_numeric_features(fieldnames, rows)
    if feature_x not in features:
        raise ScatterPlotError(f"unknown numerical feature: {feature_x}")
    if feature_y not in features:
        raise ScatterPlotError(f"unknown numerical feature: {feature_y}")

    pairs = feature_pairs(rows, feature_x, feature_y)
    if not pairs:
        raise ScatterPlotError(f"no shared numerical values for {feature_x} and {feature_y}")
    return pairs


def display_scatter_plot(pairs, feature_x, feature_y, correlation=None, output_path=None, show=True):
    plt.figure(figsize=(10, 7))

    houses = sorted({house for _, _, house in pairs if house})
    for house in houses:
        x_values = []
        y_values = []
        for x, y, pair_house in pairs:
            if pair_house == house:
                x_values.append(x)
                y_values.append(y)

        plt.scatter(
            x_values,
            y_values,
            s=24,
            alpha=0.65,
            label=house,
            color=HOUSE_COLORS.get(house),
            edgecolors="none",
        )

    if not houses:
        x_values = [x for x, _, _ in pairs]
        y_values = [y for _, y, _ in pairs]
        plt.scatter(x_values, y_values, s=24, alpha=0.65, edgecolors="none")

    title = f"{feature_x} vs {feature_y}"
    if correlation is not None:
        title += f" (correlation: {correlation:.6f})"
    plt.title(title)
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    if houses:
        plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Display a scatter plot comparing two numerical features."
    )
    parser.add_argument("dataset", help="CSV dataset path")
    parser.add_argument("feature_x", nargs="?", help="optional x-axis feature")
    parser.add_argument("feature_y", nargs="?", help="optional y-axis feature")
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="save the scatter plot image to PATH",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open the matplotlib window",
    )
    return parser.parse_args()


def main():
    configure_logger()
    args = parse_arguments()

    if (args.feature_x is None) != (args.feature_y is None):
        LOGGER.error("provide either both features or no feature")
        return 1

    try:
        if args.feature_x and args.feature_y:
            feature_x = args.feature_x
            feature_y = args.feature_y
            pairs = get_feature_pair(args.dataset, feature_x, feature_y)
            correlation = pearson_correlation(pairs)
        else:
            (
                feature_x,
                feature_y,
                correlation,
                pairs,
            ) = find_most_similar_features(args.dataset)

        LOGGER.info("most similar features: %s and %s", feature_x, feature_y)
        LOGGER.info("correlation: %.6f", correlation)
        display_scatter_plot(
            pairs,
            feature_x,
            feature_y,
            correlation=correlation,
            output_path=args.save,
            show=not args.no_show,
        )
    except ScatterPlotError as error:
        LOGGER.error("%s", error)
        return 1
    except OSError as error:
        LOGGER.error("%s", error)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
