import json
import csv
from pathlib import Path
import argparse
from src.data_visualization.pair_plot import parse_float
from src.logistic_regression.logreg_train import apply_imputation, apply_standardization, HOUSES, compute_z, sigmoid

def save_predictions(indexes, predictions, output_path) :
	if len(indexes) != len(predictions):
		raise ValueError("indexes and predictions must have the same length")

	path = Path(output_path)
	if str(path.parent) != ".":
		path.parent.mkdir(parents=True, exist_ok=True)

	with path.open("w", newline="") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(["Index", "Hogwarts House"])
		for index in range(len(predictions)):
			writer.writerow([indexes[index], predictions[index]])

def predict_houses(X, model) :
	predictions = []
	for student_values in X :
		predictions.append(predict_house(model, student_values))
	return predictions

def predict_house(model, student_values) :
	pred_temp = 0.0
	house_pred = ""

	for house in model["houses"] :
		weights = model["weights"][house]
		bias = model["biases"][house]
		z = compute_z(weights, student_values, bias)
		prediction = sigmoid(z)
		if prediction > pred_temp :
			pred_temp = prediction
			house_pred = house
	return house_pred

def load_prediction_rows(csv_path) :
	path = Path(csv_path)
	if not path.exists():
		raise FileNotFoundError(f"file not found: {csv_path}")
	if not path.is_file():
		raise ValueError(f"not a file: {csv_path}")

	with path.open(newline="") as csv_file:
		reader = csv.DictReader(csv_file)
		return list(reader)

def prepare_prediction_data(csv_path, model) :
	features = model["preprocessing_params"]["features"]
	imputation_means = model["preprocessing_params"]["imputation_means"]
	means = model["preprocessing_params"]["standardization_means"]
	stds = model["preprocessing_params"]["standardization_stds"]
	rows = load_prediction_rows(csv_path)
	X = []
	indexes = []

	for row in rows :
		student_values = []
		indexes.append(row["Index"])
		for feature in features :
			student_values.append(parse_float(row[feature]))
		X.append(student_values)
	X = apply_imputation(X, imputation_means, features)
	X_standardized = apply_standardization(X, means, stds, features)
	return X_standardized, indexes

def load_model(model_path) :
	with open(model_path, "r") as file :
		model = json.load(file)
	return model

def looks_like_dataset(path) :
	return str(path).lower().endswith(".csv")


def looks_like_weights(path) :
	return str(path).lower().endswith(".json")

def parse_arguments() :
	parser = argparse.ArgumentParser(description="Predict Hogwarts houses with a trained logistic regression model.")
	parser.add_argument("dataset", help="CSV test dataset path")
	parser.add_argument("weights", nargs="?", default="models/weights.json", help="trained weights JSON path, default: models/weights.json")
	parser.add_argument("--output", default="houses.csv", help="prediction output CSV path, default: houses.csv")
	args = parser.parse_args()

	if looks_like_weights(args.dataset) and looks_like_dataset(args.weights):
		args.dataset, args.weights = args.weights, args.dataset
	return args


def main() :
	args = parse_arguments()
	try:
		model = load_model(args.weights)
	except FileNotFoundError:
		print(f"error: weights file not found: {args.weights}")
		return 1
	except json.JSONDecodeError:
		print(f"error: invalid JSON weights file: {args.weights}")
		return 1

	try:
		X, indexes = prepare_prediction_data(args.dataset, model)
		predictions = predict_houses(X, model)
		save_predictions(indexes, predictions, args.output)
	except (FileNotFoundError, ValueError, KeyError) as error:
		print(f"error: {error}")
		return 1

	print(f"loaded model for {len(model['houses'])} houses")
	print(f"saved {len(predictions)} predictions to {args.output}")
	return 0
if __name__ == "__main__" :
	raise SystemExit(main())
