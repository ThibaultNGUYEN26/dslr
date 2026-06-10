import csv
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlencode

from flask import Flask, jsonify, request as flask_request, send_file


ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data_visualization.histogram import (
    HistogramError,
    detect_numeric_features,
    group_feature_by_house,
    load_rows,
    similarity_score,
)
from src.data_visualization.scatter_plot import (
    ScatterPlotError,
    feature_pairs,
    find_most_similar_features,
    pearson_correlation,
)


DATASET_PATH = ROOT_DIR / "datasets" / "dataset_train.csv"
DATASETS = {
    "train": ROOT_DIR / "datasets" / "dataset_train.csv",
    "test": ROOT_DIR / "datasets" / "dataset_test.csv",
}
PLOT_DIR = ROOT_DIR / "outputs" / "web_plots"


app = Flask(__name__)


def feature_slug(feature):
    slug = []
    for char in feature.lower():
        if char.isalnum():
            slug.append(char)
        else:
            slug.append("_")
    return "".join(slug).strip("_")


def numerical_features():
    fieldnames, rows = load_rows(DATASET_PATH)
    return detect_numeric_features(fieldnames, rows)


def histogram_features():
    fieldnames, rows = load_rows(DATASET_PATH)
    features = detect_numeric_features(fieldnames, rows)
    items = []

    for feature in features:
        grouped = group_feature_by_house(rows, feature)
        items.append(
            {
                "name": feature,
                "similarityScore": similarity_score(grouped, 20),
                "image": f"/api/histograms/{feature_slug(feature)}.png",
            }
        )

    return items


def scatter_plot_pairs():
    fieldnames, rows = load_rows(DATASET_PATH)
    features = detect_numeric_features(fieldnames, rows)
    pairs = []

    for index, feature_x in enumerate(features):
        for feature_y in features[index + 1 :]:
            values = feature_pairs(rows, feature_x, feature_y)
            correlation = pearson_correlation(values)
            if correlation is None:
                continue
            pairs.append(
                {
                    "featureX": feature_x,
                    "featureY": feature_y,
                    "correlation": correlation,
                    "image": "/api/scatter-plot.png?"
                    + urlencode({"x": feature_x, "y": feature_y}),
                }
            )

    if not pairs:
        raise ScatterPlotError("could not compare numerical feature pairs")
    pairs.sort(key=lambda pair: abs(pair["correlation"]), reverse=True)
    return pairs


def read_dataset(dataset_name):
    path = DATASETS.get(dataset_name)
    if path is None:
        return None, None, None
    with path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            return path, [], []
        return path, reader.fieldnames, list(reader)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.get("/api/datasets")
def datasets():
    items = []
    for name, path in DATASETS.items():
        _, columns, rows = read_dataset(name)
        items.append(
            {
                "name": name,
                "path": str(path.relative_to(ROOT_DIR)),
                "columns": columns,
                "rowCount": len(rows),
            }
        )
    return jsonify({"datasets": items})


@app.get("/api/datasets/<dataset_name>")
def dataset_rows(dataset_name):
    path, columns, rows = read_dataset(dataset_name)
    if path is None:
        return jsonify({"error": f"unknown dataset: {dataset_name}"}), 404

    query = flask_request.args.get("q", "").strip().lower()
    try:
        limit = int(flask_request.args.get("limit", "50"))
        offset = int(flask_request.args.get("offset", "0"))
    except ValueError:
        return jsonify({"error": "limit and offset must be integers"}), 400

    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    if query:
        rows = [
            row
            for row in rows
            if any(query in str(value).lower() for value in row.values())
        ]

    total = len(rows)
    page_rows = rows[offset : offset + limit]
    return jsonify(
        {
            "name": dataset_name,
            "path": str(path.relative_to(ROOT_DIR)),
            "columns": columns,
            "rows": page_rows,
            "limit": limit,
            "offset": offset,
            "total": total,
        }
    )


@app.get("/api/histograms")
def histograms():
    try:
        features = histogram_features()
    except HistogramError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(
        {
            "dataset": str(DATASET_PATH.relative_to(ROOT_DIR)),
            "features": features,
        }
    )


@app.get("/api/histograms/<slug>.png")
def histogram_image(slug):
    try:
        features = numerical_features()
    except HistogramError as error:
        return jsonify({"error": str(error)}), 400

    feature = None
    for candidate in features:
        if feature_slug(candidate) == slug:
            feature = candidate
            break

    if feature is None:
        return jsonify({"error": f"unknown histogram: {slug}"}), 404

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / f"{slug}.png"

    command = [
        sys.executable,
        str(ROOT_DIR / "src" / "data_visualization" / "histogram.py"),
        str(DATASET_PATH),
        feature,
        "--no-show",
        "--save",
        str(output_path),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "histogram failed"
        return jsonify({"error": message}), 500

    return send_file(output_path, mimetype="image/png")


@app.get("/api/pair-plot")
def pair_plot():
    return jsonify(
        {
            "dataset": str(DATASET_PATH.relative_to(ROOT_DIR)),
            "defaultImage": "/api/pair-plot.png",
            "allFeaturesImage": "/api/pair-plot.png?all=1",
        }
    )


@app.get("/api/pair-plot.png")
def pair_plot_image():
    use_all = "all" in flask_request.args
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PLOT_DIR / ("pair_plot_all.png" if use_all else "pair_plot.png")

    command = [
        sys.executable,
        str(ROOT_DIR / "src" / "data_visualization" / "pair_plot.py"),
        str(DATASET_PATH),
        "--no-show",
        "--save",
        str(output_path),
    ]
    if use_all:
        command.insert(4, "--all")

    result = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "pair plot failed"
        return jsonify({"error": message}), 500

    return send_file(output_path, mimetype="image/png")


@app.get("/api/scatter-plot")
def scatter_plot():
    try:
        feature_x, feature_y, correlation, _ = find_most_similar_features(DATASET_PATH)
        features = numerical_features()
        pairs = scatter_plot_pairs()
    except (HistogramError, ScatterPlotError) as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(
        {
            "dataset": str(DATASET_PATH.relative_to(ROOT_DIR)),
            "features": features,
            "defaultFeatureX": feature_x,
            "defaultFeatureY": feature_y,
            "correlation": correlation,
            "image": "/api/scatter-plot.png",
            "pairs": pairs,
        }
    )


@app.get("/api/scatter-plot.png")
def scatter_plot_image():
    feature_x = flask_request.args.get("x", "").strip()
    feature_y = flask_request.args.get("y", "").strip()
    if (feature_x == "") != (feature_y == ""):
        return jsonify({"error": "provide either both features or no feature"}), 400

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    if feature_x and feature_y:
        output_path = PLOT_DIR / f"scatter_{feature_slug(feature_x)}_{feature_slug(feature_y)}.png"
    else:
        output_path = PLOT_DIR / "scatter_plot.png"

    command = [
        sys.executable,
        str(ROOT_DIR / "src" / "data_visualization" / "scatter_plot.py"),
        str(DATASET_PATH),
    ]
    if feature_x and feature_y:
        command.extend([feature_x, feature_y])
    command.extend(["--no-show", "--save", str(output_path)])

    result = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip() or "scatter plot failed"
        return jsonify({"error": message}), 500

    return send_file(output_path, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
