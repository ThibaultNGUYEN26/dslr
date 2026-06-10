import csv
import math
from pathlib import Path

from src.data_analysis.statistics import describe_values


STAT_NAMES = ("Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max")


class DescribeError(Exception):
    pass


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


def load_numeric_columns(csv_path):
    path = Path(csv_path)
    if not path.exists():
        raise DescribeError(f"file not found: {csv_path}")
    if not path.is_file():
        raise DescribeError(f"not a file: {csv_path}")

    with path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise DescribeError("empty dataset") from exc

        columns = []
        for name in header:
            columns.append({"name": name, "values": []})

        for row in reader:
            for index, cell in enumerate(row):
                if index >= len(columns):
                    continue
                number = parse_float(cell)
                if number is not None:
                    columns[index]["values"].append(number)

    numeric_columns = []
    for column in columns:
        if column["values"]:
            numeric_columns.append(column)

    if not numeric_columns:
        raise DescribeError("dataset has no numerical features")
    return numeric_columns


def build_describe_table(csv_path):
    numeric_columns = load_numeric_columns(csv_path)
    table = {}
    for column in numeric_columns:
        table[column["name"]] = describe_values(column["values"])
    return table


def format_value(value):
    if value is None:
        return ""
    return f"{value:.6f}"


def format_describe_table(table):
    column_names = list(table.keys())
    column_widths = {}

    for column_name in column_names:
        width = len(column_name)
        for stat_name in STAT_NAMES:
            value_width = len(format_value(table[column_name][stat_name]))
            if value_width > width:
                width = value_width
        column_widths[column_name] = width

    row_label_width = 8
    lines = []
    header = " " * row_label_width
    for column_name in column_names:
        header += f" {column_name:>{column_widths[column_name]}}"
    lines.append(header)

    for stat_name in STAT_NAMES:
        line = f"{stat_name:<{row_label_width}}"
        for column_name in column_names:
            value = format_value(table[column_name][stat_name])
            line += f" {value:>{column_widths[column_name]}}"
        lines.append(line)

    return "\n".join(lines)


def describe(csv_path):
    table = build_describe_table(csv_path)
    return format_describe_table(table)
