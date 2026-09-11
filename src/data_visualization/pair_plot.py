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
DEFAULT_FEATURES = [
    "Astronomy",
    "Herbology",
    "Divination",
    "Muggle Studies",
    "Ancient Runes",
    "History of Magic",
    "Transfiguration",
    "Potions",
    "Charms",
    "Flying",
]
HOUSE_COLORS = {
    "Gryffindor": "#b31b1b",
    "Hufflepuff": "#d6a600",
    "Ravenclaw": "#1f4e8c",
    "Slytherin": "#1f7a4d",
}


class PairPlotError(Exception):
    pass


LOGGER = logging.getLogger("dslr.pair_plot")


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
        raise PairPlotError(f"file not found: {csv_path}")
    if not path.is_file():
        raise PairPlotError(f"not a file: {csv_path}")

    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise PairPlotError("empty dataset")
        if HOUSE_COLUMN not in reader.fieldnames:
            raise PairPlotError(f"missing column: {HOUSE_COLUMN}")
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
        raise PairPlotError("dataset needs at least two numerical features")
    return features


def select_default_features(numeric_features):
    selected = []
    for feature in DEFAULT_FEATURES:
        if feature in numeric_features:
            selected.append(feature)

    if len(selected) < 2:
        return numeric_features
    return selected


def validate_features(requested_features, numeric_features):
    selected = []
    for feature in requested_features:
        if feature not in numeric_features:
            raise PairPlotError(f"unknown numerical feature: {feature}")
        selected.append(feature)

    if len(selected) < 2:
        raise PairPlotError("pair plot needs at least two selected features")
    return selected


def collect_plot_data(rows, features):
    data = {}
    houses = []

    for row in rows:
        house = row.get(HOUSE_COLUMN, "").strip()
        if house and house not in houses:
            houses.append(house)

    for feature in features:
        data[feature] = {}
        for house in houses:
            data[feature][house] = []

    for row in rows:
        house = row.get(HOUSE_COLUMN, "").strip()
        if not house:
            continue
        for feature in features:
            value = parse_float(row.get(feature, ""))
            if value is not None:
                data[feature][house].append(value)

    return sorted(houses), data


def paired_values(rows, feature_x, feature_y, house):
    x_values = []
    y_values = []
    for row in rows:
        if row.get(HOUSE_COLUMN, "").strip() != house:
            continue

        x = parse_float(row.get(feature_x, ""))
        y = parse_float(row.get(feature_y, ""))
        if x is not None and y is not None:
            x_values.append(x)
            y_values.append(y)
    return x_values, y_values


def values_range(values):
    if not values:
        return None, None

    minimum = values[0]
    maximum = values[0]
    for value in values:
        if value < minimum:
            minimum = value
        if value > maximum:
            maximum = value
    return minimum, maximum


def histogram_density(values, minimum, maximum, bins):
    counts = [0] * bins
    if not values:
        return counts
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


def prepare_pair_plot(csv_path, requested_features=None, use_all=False):
    fieldnames, rows = load_rows(csv_path)
    numeric_features = detect_numeric_features(fieldnames, rows)

    if requested_features:
        features = validate_features(requested_features, numeric_features)
    elif use_all:
        features = numeric_features
    else:
        features = select_default_features(numeric_features)

    houses, data = collect_plot_data(rows, features)
    if not houses:
        raise PairPlotError("dataset has no house labels")

    return rows, houses, features, data


def display_pair_plot(rows, houses, features, data, output_path=None, show=True):
    size = len(features)
    figure_width = max(10, size * 1.7)
    figure_height = max(10, size * 1.7)
    figure, axes = plt.subplots(size, size, figsize=(figure_width, figure_height))

    for row_index, feature_y in enumerate(features):
        for column_index, feature_x in enumerate(features):
            axis = axes[row_index][column_index]

            if row_index == column_index:
                all_values = []
                for house in houses:
                    all_values.extend(data[feature_x][house])
                minimum, maximum = values_range(all_values)
                if minimum is None:
                    continue
                bin_count = 16
                if maximum == minimum:
                    bin_width = 1.0
                else:
                    bin_width = (maximum - minimum) / bin_count
                bin_edges = []
                for index in range(bin_count):
                    bin_edges.append(minimum + index * bin_width)

                for house in houses:
                    densities = histogram_density(
                        data[feature_x][house], minimum, maximum, bin_count
                    )
                    axis.bar(
                        bin_edges,
                        densities,
                        width=bin_width,
                        align="edge",
                        alpha=0.4,
                        color=HOUSE_COLORS.get(house),
                        edgecolor="none",
                    )
            else:
                for house in houses:
                    x_values, y_values = paired_values(rows, feature_x, feature_y, house)
                    axis.scatter(
                        x_values,
                        y_values,
                        s=5,
                        alpha=0.45,
                        color=HOUSE_COLORS.get(house),
                        edgecolors="none",
                    )

            if row_index == size - 1:
                axis.set_xlabel(feature_x, fontsize=7)
            else:
                axis.set_xticklabels([])

            if column_index == 0:
                axis.set_ylabel(feature_y, fontsize=7)
            else:
                axis.set_yticklabels([])

            axis.tick_params(axis="both", labelsize=6)

    handles = []
    labels = []
    for house in houses:
        handle = plt.Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=HOUSE_COLORS.get(house),
            label=house,
        )
        handles.append(handle)
        labels.append(house)

    figure.legend(handles, labels, loc="upper right")
    figure.suptitle("Hogwarts course pair plot", y=0.995)
    figure.tight_layout(rect=(0, 0, 0.97, 0.98))

    if output_path:
        plt.savefig(output_path)
    if show:
        plt.show()
    plt.close()


def parse_feature_list(raw_features):
    if raw_features is None:
        return None

    features = []
    for feature in raw_features.split(","):
        feature = feature.strip()
        if feature:
            features.append(feature)
    return features


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Display a scatter plot matrix for Hogwarts course features."
    )
    parser.add_argument("dataset", help="CSV dataset path")
    parser.add_argument(
        "--features",
        help="comma-separated feature list; default uses selected logistic-regression features",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="plot every numerical feature except Index",
    )
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="save the pair plot image to PATH",
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

    if args.features and args.all:
        LOGGER.error("use either --features or --all, not both")
        return 1

    try:
        requested_features = parse_feature_list(args.features)
        rows, houses, features, data = prepare_pair_plot(
            args.dataset,
            requested_features=requested_features,
            use_all=args.all,
        )

        LOGGER.info("plotting %d features", len(features))
        LOGGER.info("features: %s", ", ".join(features))
        if not args.features and not args.all:
            LOGGER.info(
                "default selection excludes Index, homogeneous courses, and redundant Defense Against the Dark Arts"
            )
            LOGGER.info("logistic regression feature proposal: %s", ", ".join(DEFAULT_FEATURES))

        display_pair_plot(
            rows,
            houses,
            features,
            data,
            output_path=args.save,
            show=not args.no_show,
        )
    except PairPlotError as error:
        LOGGER.error("%s", error)
        return 1
    except OSError as error:
        LOGGER.error("%s", error)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
