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


class HistogramError(Exception):
    pass


LOGGER = logging.getLogger("dslr.histogram")


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
        raise HistogramError(f"file not found: {csv_path}")
    if not path.is_file():
        raise HistogramError(f"not a file: {csv_path}")

    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise HistogramError("empty dataset")
        if HOUSE_COLUMN not in reader.fieldnames:
            raise HistogramError(f"missing column: {HOUSE_COLUMN}")
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

    if not features:
        raise HistogramError("dataset has no numerical features")
    return features


def group_feature_by_house(rows, feature):
    grouped = {}
    for row in rows:
        house = row.get(HOUSE_COLUMN, "").strip()
        value = parse_float(row.get(feature, ""))
        if not house or value is None:
            continue
        if house not in grouped:
            grouped[house] = []
        grouped[house].append(value)

    if len(grouped) < 2:
        raise HistogramError(f"not enough houses with values for feature: {feature}")
    return grouped


def normalized_histogram(values, minimum, maximum, bins):
    counts = [0] * bins
    if maximum == minimum:
        return counts

    for value in values:
        index = int((value - minimum) / (maximum - minimum) * bins)
        if index == bins:
            index -= 1
        counts[index] += 1

    total = 0
    for count in counts:
        total += count
    if total == 0:
        return counts

    return [count / total for count in counts]


def values_range(grouped_values):
    first_value = None
    for values in grouped_values.values():
        for value in values:
            first_value = value
            break
        if first_value is not None:
            break

    if first_value is None:
        raise HistogramError("feature has no numerical values")

    minimum = first_value
    maximum = first_value
    for values in grouped_values.values():
        for value in values:
            if value < minimum:
                minimum = value
            if value > maximum:
                maximum = value
    return minimum, maximum


def histogram_density(values, minimum, maximum, bins):
    counts = [0] * bins
    if maximum == minimum:
        counts[0] = len(values)
        return counts

    for value in values:
        index = int((value - minimum) / (maximum - minimum) * bins)
        if index == bins:
            index -= 1
        counts[index] += 1

    total = 0
    for count in counts:
        total += count
    if total == 0:
        return counts

    bin_width = (maximum - minimum) / bins
    return [count / (total * bin_width) for count in counts]


def similarity_score(grouped_values, bins):
    minimum, maximum = values_range(grouped_values)

    histograms = []
    for values in grouped_values.values():
        histograms.append(normalized_histogram(values, minimum, maximum, bins))

    score = 0.0
    for bin_index in range(bins):
        mean = 0.0
        for histogram in histograms:
            mean += histogram[bin_index]
        mean /= len(histograms)

        variance = 0.0
        for histogram in histograms:
            variance += (histogram[bin_index] - mean) ** 2
        score += variance / len(histograms)

    return score


def find_most_homogeneous_feature(csv_path, bins=20):
    fieldnames, rows = load_rows(csv_path)
    features = detect_numeric_features(fieldnames, rows)

    best_feature = None
    best_score = None
    best_grouped = None

    for feature in features:
        grouped = group_feature_by_house(rows, feature)
        score = similarity_score(grouped, bins)
        if best_score is None or score < best_score:
            best_feature = feature
            best_score = score
            best_grouped = grouped

    return best_feature, best_score, best_grouped


def get_feature_distribution(csv_path, feature):
    fieldnames, rows = load_rows(csv_path)
    features = detect_numeric_features(fieldnames, rows)
    if feature not in features:
        raise HistogramError(f"unknown numerical feature: {feature}")
    return group_feature_by_house(rows, feature)


def display_histogram(grouped_values, feature, bins=20, output_path=None, show=True):
    plt.figure(figsize=(11, 7))
    minimum, maximum = values_range(grouped_values)
    if maximum == minimum:
        bin_width = 1.0
    else:
        bin_width = (maximum - minimum) / bins

    bin_edges = []
    for index in range(bins):
        bin_edges.append(minimum + index * bin_width)

    for house in sorted(grouped_values):
        values = grouped_values[house]
        densities = histogram_density(values, minimum, maximum, bins)
        plt.bar(
            bin_edges,
            densities,
            width=bin_width,
            align="edge",
            alpha=0.45,
            label=house,
            color=HOUSE_COLORS.get(house),
            edgecolor="black",
            linewidth=0.4,
        )

    plt.title(f"{feature} score distribution by Hogwarts house")
    plt.xlabel(feature)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Display a histogram of Hogwarts course scores by house."
    )
    parser.add_argument("dataset", help="CSV dataset path")
    parser.add_argument(
        "feature",
        nargs="?",
        help="optional numerical feature to display instead of auto-selecting",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=20,
        help="number of histogram bins, default: 20",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="save the histogram image to PATH",
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

    if args.bins <= 0:
        LOGGER.error("bins must be greater than 0")
        return 1

    try:
        if args.feature:
            feature = args.feature
            score = None
            grouped = get_feature_distribution(args.dataset, feature)
        else:
            feature, score, grouped = find_most_homogeneous_feature(
                args.dataset, args.bins
            )

        if score is None:
            LOGGER.info("displaying feature: %s", feature)
        else:
            LOGGER.info("most homogeneous course: %s", feature)
            LOGGER.info("distribution similarity score: %.6f", score)

        display_histogram(
            grouped,
            feature,
            bins=args.bins,
            output_path=args.save,
            show=not args.no_show,
        )
    except HistogramError as error:
        LOGGER.error("%s", error)
        return 1
    except OSError as error:
        LOGGER.error("%s", error)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
