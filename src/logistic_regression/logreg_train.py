import argparse
import math
import json
from src.data_visualization.pair_plot import load_rows ,parse_float ,DEFAULT_FEATURES
from src.data_analysis.statistics import ft_mean, ft_std

HOUSES = ["Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin"]

def save_model(model) :
	output_path = "models/weights.json"
	with open(output_path, "w") as file :
		json.dump(model, file, indent=2)

def update_parameters(weights, bias, gradient_weights, gradient_bias, lr) :
	for index in range(len(weights)) :
		weights[index] = weights[index] - lr * gradient_weights[index]
	bias = bias - lr * gradient_bias
	return weights, bias

def average_gradients(gradient_weights, gradient_bias, batch_size) :
	if batch_size == 0:
		raise ValueError("cannot average gradients with an empty batch")

	for index in range(len(gradient_weights)) :
		gradient_weights[index] = gradient_weights[index] / batch_size
	gradient_bias = gradient_bias / batch_size
	return gradient_weights, gradient_bias

def gradient_descent(X, y, weights, bias, lr) :
	gradient_weights = [0.0] * len(weights)
	gradient_bias = 0.0
	batch_size = len(X)

	for index in range(batch_size) :
		student_value = X[index]
		true_label = y[index]
		z = compute_z(weights, student_value, bias)
		prediction = sigmoid(z)
		error = prediction - true_label
		for feature in range(len(X[index])) :
			gradient_weights[feature] += error * student_value[feature]
		gradient_bias += error
	gradient_weights, gradient_bias = average_gradients(gradient_weights, gradient_bias, batch_size)
	weights, bias = update_parameters(weights, bias, gradient_weights, gradient_bias, lr)
	return weights, bias

def train_all_houses(X, y, training_config) :
	model = {
		"houses": HOUSES,
		"weights": {},
		"biases": {}
	}

	for house in HOUSES :
		weights, bias = train_one_house(X, y, house, training_config)
		model["weights"][house] = weights
		model["biases"][house] = bias
	return model

def train_one_house(X, y, target_house, training_config):
	learning_rate = training_config["learning_rate"]
	epochs = training_config["epochs"]
	batch_size = training_config["batch_size"]
	batch = [1, 32, len(X)]

	if batch_size is None :
		batch_size = len(X)

	if epochs <= 0 :
		raise ValueError("epochs must be greater than 0")
	if learning_rate <= 0 :
		raise ValueError("learning_rate must be greater than 0")
	if batch_size not in batch :
		raise ValueError("batch_size must be 1, 32, or the full training size")

	y_binary = one_vs_all_labels(target_house, y)
	weights, bias = initialize_model(X)

	for epoch in range(epochs) :
		for start in range(0, len(X), batch_size) :
			end = start + batch_size
			batch_X = X[start:end]
			batch_y = y_binary[start:end]
			weights, bias = gradient_descent(batch_X, batch_y, weights, bias, learning_rate)
	return weights, bias


def sigmoid(z):
	# Sigmoid: transforms a score z into a value between 0 and 1.
	# Uses an equivalent formula for negative z to avoid overflow.
	if z >= 0:
		return 1 / (1 + math.exp(-z))
	exp_z = math.exp(z)
	return exp_z / (1 + exp_z)

def initialize_model(X) :
	bias = 0.0
	if len(X) != 0 :
		weights = [0.0] * len(X[0])
	else :
		raise ValueError("cannot initialize model with empty training data")
	return weights, bias

def one_vs_all_labels(target_house, y) :
	if target_house not in y:
		raise ValueError(f"unknown target house: {target_house}")
	binary_labels = []
	for house in y :
		if target_house == house :
			binary_labels.append(1)
		else :
			binary_labels.append(0)
	return binary_labels

def compute_z(weights, student_values, bias) :
	# z = bias + sum(weight_i * student_value_i)
	z = bias
	if len(student_values) != len(weights):
		raise ValueError("student_values and weights must have the same length")
	for index in range(len(student_values)):
		value = student_values[index]
		weight = weights[index]
		z += weight * value
	return z	

def apply_standardization(X, means, stds, features) :
	# valeur_standardisée = (valeur - moyenne_de_la_feature) / écart_type_de_la_feature
	for row in X :
		for index in range(len(row)) :
			value = row[index]
			feature = features[index]
			if stds[feature] == 0:
				row[index] = 0
			else :
				row[index] = (value - means[feature]) / stds[feature]
	return X

def compute_standardization_params(X, features) :
	means = {}
	stds = {}

	for index in range(len(features)) :
		feature = features[index]
		column_values = []

		for row in X :
			value = row[index]
			column_values.append(value)
		means[feature] = ft_mean(column_values)
		stds[feature] = ft_std(column_values)
	return means, stds

def apply_imputation(X, imputation_means, features) :
	for row in X :
		for index in range(len(row)) :
			if row[index] is None:
				feature_name = features[index]
				row[index] = imputation_means[feature_name]
	return X

def compute_imputation_means(X, features) :
	imputation_means = {}

	for index in range(len(features)) :
		feature = features[index]
		column_values = []

		for row in X :
			value = row[index]
			if value is not None :
				column_values.append(value)

		imputation_means[feature] = ft_mean(column_values)
	return imputation_means

def prepare_training_data(csv_path) :
	X = [] 
	y = []
	preprocessing_params = {}
	features = DEFAULT_FEATURES

	fieldnames, rows = load_rows(csv_path)
	for row in rows :
		student_values = []
		for feature in features :
			student_values.append(parse_float(row[feature]))
		y.append(row["Hogwarts House"])
		X.append(student_values)

	imputation_means = compute_imputation_means(X, features)
	X = apply_imputation(X, imputation_means, features)
	means, stds = compute_standardization_params(X, features)

	preprocessing_params["features"] = features
	preprocessing_params["imputation_means"] = imputation_means
	preprocessing_params["standardization_means"] = means
	preprocessing_params["standardization_stds"] = stds

	X_standardized = apply_standardization(X, means, stds, features)

	return X_standardized, y, preprocessing_params

def parse_arguments() :
	parser = argparse.ArgumentParser(description="Train a logistic regression model.")
	parser.add_argument("dataset", help="CSV training dataset path")
	parser.add_argument("--learning-rate", type=float, default=0.01, help="learning rate, default: 0.01")
	parser.add_argument("--epochs", type=int, default=1000, help="number of training epochs, default: 1000")
	parser.add_argument("--batch-size", type=int, default=None, help="mini-batch size; default uses the full dataset")
	return parser.parse_args()

def main() :
	args = parse_arguments()

	training_config = {
		"learning_rate" : args.learning_rate,
		"epochs" : args.epochs,
		"batch_size" : args.batch_size,
	}

	try:
		X, y, preprocessing_params = prepare_training_data(args.dataset)
		model = train_all_houses(X, y, training_config)
		model["preprocessing_params"] = preprocessing_params
		model["training_config"] = training_config
		save_model(model)
	except (FileNotFoundError, ValueError, KeyError) as error:
		print(f"error: {error}")
		return 1

	print(f"loaded {len(X)} students")
	print(f"features: {len(preprocessing_params['features'])}")
	print(f"learning_rate: {args.learning_rate}")
	print(f"epochs: {args.epochs}")
	print(f"batch_size: {args.batch_size or len(X)}")
	return 0

if __name__ == "__main__" :
	raise SystemExit(main())
