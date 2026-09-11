# DSLR

This project is part of the **Data Science × Logistic Regression** subject.

The goal of this first mandatory part is to explore a dataset and reproduce a simplified version of the behavior of `pandas.describe()` without using built-in statistical helper functions.

## Data Analysis

### Describe

The program `describe.[extension]` takes a dataset as input and displays basic statistical information for every numerical feature.

The objective is to better understand the raw data before using it for machine learning.
This step allows us to inspect the structure of the dataset, identify numerical columns, and detect potential anomalies.

#### Usage

```bash
./describe.[extension] dataset_train.csv
```

Example:

```bash
./describe.py dataset_train.csv
```

#### Expected Output

For each numerical feature, the program displays:

```text
             Feature 1   Feature 2   Feature 3   Feature 4
Count        149.000000  149.000000  149.000000  149.000000
Mean         5.848322    3.051007    3.774497    1.205369
Std          5.906338    3.081445    4.162021    1.424286
Min          4.300000    2.000000    1.000000    0.100000
25%          5.100000    2.800000    1.600000    0.300000
50%          5.800000    3.000000    4.400000    1.300000
75%          6.400000    3.300000    5.100000    1.800000
Max          7.900000    4.400000    6.900000    2.500000
```

#### Statistics Computed

The program computes the following values for each numerical column:

##### Count

The number of valid numerical values in the feature.

##### Mean

The average value of the feature.

Formula:

```text
mean = sum(values) / count
```

##### Standard Deviation

The standard deviation measures how spread out the values are around the mean.

Formula:

```text
std = sqrt(sum((value - mean)^2) / count)
```

##### Minimum

The smallest value of the feature.

##### Percentiles

The program computes the following percentiles:

* 25%
* 50%
* 75%

The 50% percentile is also known as the median.

##### Maximum

The largest value of the feature.

#### Constraints

The following functions are forbidden:

* `count`
* `mean`
* `std`
* `min`
* `max`
* `percentile`
* `describe`
* Any library function that directly performs the required statistical computation

All statistics must be implemented manually.

#### Example

```bash
make describe
python describe.py datasets/dataset_train.csv
```

Output:

```text
             Astronomy   Herbology   Defense Against the Dark Arts
Count        395.000000  395.000000  395.000000
Mean         15639.1234  1.234567    -42.987654
Std          520.456789  3.456789    15.123456
Min          12345.0000  -10.00000   -100.00000
25%          15200.0000  -1.500000   -50.000000
50%          15600.0000  1.200000    -42.000000
75%          16000.0000  4.000000    -35.000000
Max          17000.0000  12.00000    0.000000
```

## Data Visualization

The visualizations group students by Hogwarts house and use a separate color for each house. Invalid numerical values are ignored.

### Histogram

The histogram compares the distribution of one course across Gryffindor, Hufflepuff, Ravenclaw, and Slytherin. Each distribution is normalized, so houses with different numbers of students can still be compared fairly.

By default, the program examines every numerical course and selects the most homogeneous one. It compares normalized histograms on the same scale and calculates how different the house distributions are in every bin. The course with the lowest difference score has the most similar distributions between houses.

Strongly overlapping distributions indicate that the course does not distinguish the houses very well. Clearly separated distributions indicate that the course may be useful for classification.

```bash
make histogram
make histogram COURSE="Astronomy"
make histogram-save OUT=histogram.png
```

### Scatter Plot

The scatter plot places one numerical feature on each axis. Every point represents one student, and its color represents that student's Hogwarts house.

By default, the program tests every pair of numerical features and selects the pair with the highest absolute Pearson correlation. A correlation close to `1` means the two features increase together, while a correlation close to `-1` means one tends to decrease as the other increases. A value close to `0` indicates little linear relationship.

This plot helps identify redundant features and reveals whether combinations of courses form visible house clusters.

```bash
make scatter
make scatter X="Astronomy" Y="Herbology"
make scatter-save X="Astronomy" Y="Herbology" OUT=scatter_plot.png
```

### Pair Plot

The pair plot provides a matrix view of several numerical features:

* The diagonal contains histograms showing the distribution of each feature by house.
* Every other cell contains a scatter plot comparing the feature of its column with the feature of its row.

Looking at all these relationships together makes it easier to find useful classification features, house separation, correlations, outliers, and redundant courses. The default view uses the features selected for logistic regression; `pair-all` includes every numerical feature and therefore produces a much larger matrix.

```bash
make pair
make pair-all
make pair-save OUT=pair_plot.png
```

## Logistic Regression

### Training

The training program uses ten selected course features to predict a student's Hogwarts house. Numerical values are prepared using parameters calculated only from the training dataset: unavailable values are replaced with the feature mean, then every feature is standardized so courses with very different scales can be trained together.

Because the dataset contains four houses, the program uses a one-vs-all approach. It trains one binary logistic regression classifier for each house:

* Gryffindor against all other houses
* Hufflepuff against all other houses
* Ravenclaw against all other houses
* Slytherin against all other houses

For every student, the model calculates a decision score from the ten standardized features. The sigmoid function converts that score into a probability between `0` and `1`. Binary cross-entropy measures the prediction loss, and gradient descent updates the weights and bias over multiple epochs.

```bash
make train
python -m src.logistic_regression.logreg_train datasets/dataset_train.csv
```

The model uses Batch gradient descent, which updates the weights and bias once per epoch using the complete training dataset. The trained weights, biases, and preprocessing parameters are saved in `models/weights.json`.

### Prediction

The prediction program loads `models/weights.json` and applies the same feature preparation used during training. It evaluates the four one-vs-all classifiers for every test student and selects the house with the highest predicted probability.

```bash
make predict
python -m src.logistic_regression.logreg_predict datasets/dataset_test.csv models/weights.json --output houses.csv
```

The generated `houses.csv` file contains each student index and predicted Hogwarts house.

## Bonus

### Enhanced Describe

In addition to the mandatory statistics, `describe.py` computes four supplementary fields for every numerical feature.

#### Missing

The number of dataset rows that do not contain a valid numerical value for the feature. This helps identify columns that may require additional data preparation.

#### Variance

The average squared distance between the values and their mean. It measures dispersion before the square root used for standard deviation is applied.

```text
variance = sum((value - mean)^2) / count
```

#### Range

The distance between the largest and smallest values. It provides a simple measurement of the complete spread of a feature.

```text
range = maximum - minimum
```

#### Interquartile Range (IQR)

The distance between the 75th and 25th percentiles. It measures the spread of the middle half of the values and is less affected by extreme values than the complete range.

```text
IQR = 75% - 25%
```
