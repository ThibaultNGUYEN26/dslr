import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
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
from src.data_analysis.describe import (
    DescribeError,
    STAT_NAMES,
    build_describe_table,
)
from src.logistic_regression.logreg_train import sigmoid


DATASET_PATH = ROOT_DIR / "datasets" / "dataset_train.csv"
DATASETS = {
    "train": ROOT_DIR / "datasets" / "dataset_train.csv",
    "test": ROOT_DIR / "datasets" / "dataset_test.csv",
}
PLOT_DIR = ROOT_DIR / "outputs" / "web_plots"
HISTOGRAM_SCRIPT = ROOT_DIR / "src" / "data_visualization" / "histogram.py"
PAIR_PLOT_SCRIPT = ROOT_DIR / "src" / "data_visualization" / "pair_plot.py"
SCATTER_PLOT_SCRIPT = ROOT_DIR / "src" / "data_visualization" / "scatter_plot.py"
MODEL_PATH = ROOT_DIR / "models" / "weights.json"


app = Flask(__name__)


def model_updated_at():
    return datetime.fromtimestamp(
        MODEL_PATH.stat().st_mtime, timezone.utc
    ).isoformat()


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


def plot_is_current(output_path, sources):
    if not output_path.exists():
        return False

    output_mtime = output_path.stat().st_mtime
    for source in sources:
        if source.exists() and source.stat().st_mtime > output_mtime:
            return False
    return True


def generate_plot_if_needed(output_path, sources, command, error_label):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    if plot_is_current(output_path, sources):
        return None

    result = subprocess.run(
        command,
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return result.stderr.strip() or result.stdout.strip() or f"{error_label} failed"
    return None


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


@app.get("/api/describe/<dataset_name>")
def describe_dataset(dataset_name):
    path = DATASETS.get(dataset_name)
    if path is None:
        return jsonify({"error": f"unknown dataset: {dataset_name}"}), 404

    try:
        table = build_describe_table(path)
    except (DescribeError, OSError) as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(
        {
            "name": dataset_name,
            "path": str(path.relative_to(ROOT_DIR)),
            "statistics": list(STAT_NAMES),
            "features": [
                {"name": feature, "values": values}
                for feature, values in table.items()
            ],
        }
    )


@app.get("/api/training-loss")
def training_loss():
    try:
        with MODEL_PATH.open() as model_file:
            model = json.load(model_file)
    except FileNotFoundError:
        return jsonify({"error": "trained model not found"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "trained model is not valid JSON"}), 500

    histories = model.get("optimizer_loss_history", {})
    house_histories = model.get("optimizer_house_loss_history", {})
    houses = model.get("houses", [])
    strategy_names = {
        "batch": "Batch",
        "stochastic": "Stochastic",
        "mini_batch": "Mini-batch",
    }
    strategies = [strategy for strategy in strategy_names if histories.get(strategy)]
    if len(strategies) != len(strategy_names):
        return jsonify({"error": "retrain the model to compare optimizer losses"}), 409
    if not houses or any(
        not house_histories.get(strategy, {}).get(house)
        for strategy in strategies
        for house in houses
    ):
        return jsonify({"error": "retrain the model to compare per-house losses"}), 409

    epoch_count = min(
        [len(histories[strategy]) for strategy in strategies]
        + [
            len(house_histories[strategy][house])
            for strategy in strategies
            for house in houses
        ]
    )

    return jsonify(
        {
            "model": str(MODEL_PATH.relative_to(ROOT_DIR)),
            "modelUpdatedAt": model_updated_at(),
            "epochs": epoch_count,
            "trainingConfig": model.get("training_config", {}),
            "optimizerConfigs": model.get("optimizer_configs", {}),
            "series": [
                {
                    "name": strategy_names[strategy],
                    "strategy": strategy,
                    "kind": "optimizer",
                    "values": histories[strategy][:epoch_count],
                }
                for strategy in strategies
            ],
            "houseSeries": {
                strategy: [
                    {
                        "name": house,
                        "strategy": strategy,
                        "kind": "house",
                        "values": house_histories[strategy][house][:epoch_count],
                    }
                    for house in houses
                ]
                for strategy in strategies
            },
        }
    )


@app.get("/api/regression-curves")
def regression_curves():
    try:
        with MODEL_PATH.open() as model_file:
            model = json.load(model_file)
    except FileNotFoundError:
        return jsonify({"error": "trained model not found"}), 404
    except json.JSONDecodeError:
        return jsonify({"error": "trained model is not valid JSON"}), 500

    preprocessing = model.get("preprocessing_params", {})
    features = preprocessing.get("features", [])
    houses = model.get("houses", [])
    optimizer_models = model.get("optimizer_models", {})
    if not features or not houses or not optimizer_models:
        return jsonify({"error": "retrain the model to generate regression curves"}), 409

    default_house = "Hufflepuff" if "Hufflepuff" in houses else houses[0]
    house = flask_request.args.get("house", default_house).strip()
    if house not in houses:
        return jsonify({"error": f"unknown house: {house}"}), 400

    _, _, rows = read_dataset("train")
    prepared_rows = []
    for row in rows:
        student_values = []
        for feature in features:
            raw_value = row.get(feature, "").strip()
            try:
                value = float(raw_value) if raw_value else preprocessing["imputation_means"][feature]
            except ValueError:
                value = preprocessing["imputation_means"][feature]
            mean = preprocessing["standardization_means"][feature]
            std = preprocessing["standardization_stds"][feature]
            student_values.append(0.0 if std == 0 else (value - mean) / std)
        prepared_rows.append(
            (
                student_values,
                1 if row.get("Hogwarts House", "").strip() == house else 0,
            )
        )

    strategy_names = {
        "batch": "Batch",
        "stochastic": "Stochastic",
        "mini_batch": "Mini-batch",
    }
    series = []
    sample_count = 160
    for strategy, name in strategy_names.items():
        strategy_model = optimizer_models.get(strategy)
        if strategy_model is None:
            return jsonify({"error": f"missing optimizer model: {strategy}"}), 409
        weights = strategy_model["weights"][house]
        bias = strategy_model["biases"][house]
        points = []
        for student_values, label in prepared_rows:
            score = bias
            for index in range(len(weights)):
                score += weights[index] * student_values[index]
            points.append({"x": score, "y": label})

        minimum = points[0]["x"]
        maximum = points[0]["x"]
        for point in points:
            if point["x"] < minimum:
                minimum = point["x"]
            if point["x"] > maximum:
                maximum = point["x"]

        curve = []
        for index in range(sample_count):
            ratio = index / (sample_count - 1)
            x_value = minimum + ratio * (maximum - minimum)
            curve.append({"x": x_value, "y": sigmoid(x_value)})
        series.append(
            {
                "name": name,
                "strategy": strategy,
                "values": curve,
                "points": points,
            }
        )

    return jsonify(
        {
            "dataset": str(DATASET_PATH.relative_to(ROOT_DIR)),
            "model": str(MODEL_PATH.relative_to(ROOT_DIR)),
            "modelUpdatedAt": model_updated_at(),
            "featureCount": len(features),
            "house": house,
            "houses": houses,
            "series": series,
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

    output_path = PLOT_DIR / f"{slug}.png"

    command = [
        sys.executable,
        str(HISTOGRAM_SCRIPT),
        str(DATASET_PATH),
        feature,
        "--no-show",
        "--save",
        str(output_path),
    ]
    message = generate_plot_if_needed(
        output_path,
        [DATASET_PATH, HISTOGRAM_SCRIPT],
        command,
        "histogram",
    )
    if message is not None:
        return jsonify({"error": message}), 500

    return send_file(output_path, mimetype="image/png")


@app.get("/api/pair-plot")
def pair_plot():
    try:
        features = numerical_features()
    except HistogramError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(
        {
            "dataset": str(DATASET_PATH.relative_to(ROOT_DIR)),
            "features": features,
            "defaultImage": "/api/pair-plot.png",
            "allFeaturesImage": "/api/pair-plot.png?all=1",
        }
    )


@app.get("/api/pair-plot.png")
def pair_plot_image():
    use_all = "all" in flask_request.args
    raw_features = flask_request.args.get("features", "").strip()
    if use_all and raw_features:
        return jsonify({"error": "provide either all or features, not both"}), 400

    selected_features = []
    if raw_features:
        try:
            available_features = numerical_features()
        except HistogramError as error:
            return jsonify({"error": str(error)}), 400

        for feature in raw_features.split(","):
            feature = feature.strip()
            if not feature:
                continue
            if feature not in available_features:
                return jsonify({"error": f"unknown feature: {feature}"}), 400
            selected_features.append(feature)

        if len(selected_features) != 2:
            return jsonify({"error": "select exactly two features"}), 400

    if selected_features:
        slug = "_".join(feature_slug(feature) for feature in selected_features)
        output_path = PLOT_DIR / f"pair_plot_{slug}.png"
    else:
        output_path = PLOT_DIR / ("pair_plot_all.png" if use_all else "pair_plot.png")

    command = [
        sys.executable,
        str(PAIR_PLOT_SCRIPT),
        str(DATASET_PATH),
        "--no-show",
        "--save",
        str(output_path),
    ]
    if use_all:
        command.insert(4, "--all")
    if selected_features:
        command.insert(4, "--features")
        command.insert(5, ",".join(selected_features))

    message = generate_plot_if_needed(
        output_path,
        [DATASET_PATH, PAIR_PLOT_SCRIPT],
        command,
        "pair plot",
    )
    if message is not None:
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

    if feature_x and feature_y:
        output_path = PLOT_DIR / f"scatter_{feature_slug(feature_x)}_{feature_slug(feature_y)}.png"
    else:
        output_path = PLOT_DIR / "scatter_plot.png"

    command = [
        sys.executable,
        str(SCATTER_PLOT_SCRIPT),
        str(DATASET_PATH),
    ]
    if feature_x and feature_y:
        command.extend([feature_x, feature_y])
    command.extend(["--no-show", "--save", str(output_path)])

    message = generate_plot_if_needed(
        output_path,
        [DATASET_PATH, SCATTER_PLOT_SCRIPT],
        command,
        "scatter plot",
    )
    if message is not None:
        return jsonify({"error": message}), 500

    return send_file(output_path, mimetype="image/png")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
