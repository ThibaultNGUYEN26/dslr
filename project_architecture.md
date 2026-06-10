## Project Architecture

The project is divided into three main parts:

1. **Data Analysis**
2. **Data Visualization**
3. **Logistic Regression**

Each part has its own dedicated module inside the `src/` directory.
Common logic such as CSV parsing, preprocessing, formatting, and error handling is shared between the different parts of the project.

```text
dslr/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── dataset_train.csv
│   └── dataset_test.csv
│
├── describe.py
├── histogram.py
├── scatter_plot.py
├── pair_plot.py
├── logreg_train.py
├── logreg_predict.py
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_analysis/
│   │   ├── __init__.py
│   │   ├── describe.py
│   │   └── statistics.py
│   │
│   ├── data_visualization/
│   │   ├── __init__.py
│   │   ├── histogram.py
│   │   ├── scatter_plot.py
│   │   └── pair_plot.py
│   │
│   ├── logistic_regression/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── metrics.py
│   │
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   ├── scaler.py
│   │   └── encoder.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── csv_reader.py
│       ├── column_detector.py
│       ├── formatter.py
│       └── errors.py
│
├── models/
│   └── weights.json
│
└── outputs/
    ├── plots/
    └── predictions/
```

### Root Files

The root of the project contains the main executable scripts required by the subject.

| File                | Description                                                                             |
| ------------------- | --------------------------------------------------------------------------------------- |
| `describe.py`       | Runs the data analysis part and displays descriptive statistics for numerical features. |
| `histogram.py`      | Generates histograms to compare feature distributions.                                  |
| `scatter_plot.py`   | Generates scatter plots to compare relationships between features.                      |
| `pair_plot.py`      | Generates a pair plot to visualize multiple feature relationships.                      |
| `logreg_train.py`   | Trains the logistic regression model and saves the learned weights.                     |
| `logreg_predict.py` | Uses the trained model to predict Hogwarts houses from the test dataset.                |

These files act as entry points.
The actual logic is implemented inside the `src/` directory.

---

## `data/`

```text
data/
├── dataset_train.csv
└── dataset_test.csv
```

This folder contains the datasets used by the project.

* `dataset_train.csv` is used for data exploration, visualization, and model training.
* `dataset_test.csv` is used for final predictions.

---

## `src/data_analysis/`

```text
src/data_analysis/
├── describe.py
└── statistics.py
```

This module contains the logic for the **Data Analysis** part of the project.

### `describe.py`

Contains the main function used to generate descriptive statistics for every numerical feature.

It computes:

* Count
* Mean
* Standard deviation
* Minimum
* 25th percentile
* 50th percentile
* 75th percentile
* Maximum

### `statistics.py`

Contains manual implementations of statistical functions.

Examples:

* `manual_count()`
* `manual_sum()`
* `manual_mean()`
* `manual_std()`
* `manual_min()`
* `manual_max()`
* `manual_percentile()`

The subject forbids using functions such as `mean`, `std`, `min`, `max`, `percentile`, or `describe`, so these calculations are implemented manually.

---

## `src/data_visualization/`

```text
src/data_visualization/
├── histogram.py
├── scatter_plot.py
└── pair_plot.py
```

This module contains the logic for the **Data Visualization** part of the project.

### `histogram.py`

Generates histograms to analyze feature distributions and compare them between Hogwarts houses.

### `scatter_plot.py`

Generates scatter plots to observe relationships between two features.

This helps identify features that may be correlated or redundant.

### `pair_plot.py`

Generates a pair plot to visualize relationships between multiple numerical features at once.

This part helps select relevant features before training the logistic regression model.

---

## `src/logistic_regression/`

```text
src/logistic_regression/
├── model.py
├── trainer.py
├── predictor.py
└── metrics.py
```

This module contains the logic for the **Logistic Regression** part of the project.

### `model.py`

Contains the mathematical core of the logistic regression model.

Possible functions:

* `sigmoid()`
* `softmax()`
* `compute_loss()`
* `gradient_descent_step()`

### `train.py`

Handles the training process.

Responsibilities:

* Prepare the training data
* Train the model
* Update weights
* Save the trained model

### `predict.py`

Handles predictions using the trained model.

Responsibilities:

* Load saved weights
* Predict houses for the test dataset
* Generate the final prediction file

### `metrics.py`

Contains evaluation functions.

Examples:

* `accuracy()`
* `confusion_matrix()`

---

## `src/preprocessing/`

```text
src/preprocessing/
├── cleaner.py
├── scaler.py
└── encoder.py
```

This module contains shared preprocessing logic used before visualization and training.

### `cleaner.py`

Handles missing or invalid values.

Possible responsibilities:

* Remove empty values
* Replace missing numerical values
* Select useful features

### `scaler.py`

Handles feature scaling.

Possible responsibilities:

* Normalize values
* Standardize numerical features

Scaling is important for logistic regression because features can have very different ranges.

### `encoder.py`

Handles categorical encoding.

Possible responsibilities:

* Encode Hogwarts houses into numerical labels
* Decode predicted labels back into house names
* Prepare one-vs-all labels for logistic regression

Example:

```text
Gryffindor → 0
Hufflepuff → 1
Ravenclaw → 2
Slytherin → 3
```

---

## `src/utils/`

```text
src/utils/
├── csv_reader.py
├── column_detector.py
├── formatter.py
└── errors.py
```

This module contains generic helper functions reused across the project.

### `csv_reader.py`

Handles CSV file reading and writing.

Possible functions:

* `read_csv()`
* `write_csv()`

### `column_detector.py`

Detects numerical and categorical columns.

Possible functions:

* `is_float()`
* `get_numerical_columns()`
* `get_categorical_columns()`

### `formatter.py`

Handles output formatting.

For example, it formats the table displayed by `describe.py`.

### `errors.py`

Contains custom project errors.

Example:

```python
class DatasetError(Exception):
    pass
```

---

## `models/`

```text
models/
└── weights.json
```

This folder stores the trained model parameters.

For example, after running:

```bash
python logreg_train.py data/dataset_train.csv
```

The learned weights can be saved into:

```text
models/weights.json
```

These weights are then reused by `logreg_predict.py`.

---

## `outputs/`

```text
outputs/
├── plots/
└── predictions/
```

This folder stores generated files.

### `outputs/plots/`

Contains generated visualization files.

Examples:

```text
outputs/plots/histogram.png
outputs/plots/scatter_plot.png
outputs/plots/pair_plot.png
```

### `outputs/predictions/`

Contains prediction results.

Example:

```text
outputs/predictions/houses.csv
```

---

## Execution Flow

### 1. Data Analysis

```bash
python describe.py data/dataset_train.csv
```

This command reads the training dataset, detects numerical features, and displays descriptive statistics.

### 2. Data Visualization

```bash
python histogram.py data/dataset_train.csv
python scatter_plot.py data/dataset_train.csv
python pair_plot.py data/dataset_train.csv
```

These commands generate visualizations to better understand the dataset and select useful features.

### 3. Logistic Regression

```bash
python logreg_train.py data/dataset_train.csv
python logreg_predict.py data/dataset_test.csv
```

The first command trains the model and saves the weights.
The second command loads the saved weights and generates predictions for the test dataset.

---

## Design Choice

The architecture separates the project into independent modules while keeping shared logic reusable.

This makes the project easier to:

* Read
* Debug
* Test
* Extend
* Maintain

Each part of the project has a clear responsibility, and the executable files remain simple entry points that call the logic implemented in `src/`.
