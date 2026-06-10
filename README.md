# DSLR - Data Analysis

This project is part of the **Data Science × Logistic Regression** subject.

The goal of this first mandatory part is to explore a dataset and reproduce a simplified version of the behavior of `pandas.describe()` without using built-in statistical helper functions.

## V.1 Data Analysis

The program `describe.[extension]` takes a dataset as input and displays basic statistical information for every numerical feature.

The objective is to better understand the raw data before using it for machine learning.
This step allows us to inspect the structure of the dataset, identify numerical columns, observe value ranges, and detect potential anomalies or missing values.

## Usage

```bash
./describe.[extension] dataset_train.csv
```

Example:

```bash
./describe.py dataset_train.csv
```

## Expected Output

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

## Statistics Computed

The program computes the following values for each numerical column:

### Count

The number of valid numerical values in the feature.

### Mean

The average value of the feature.

Formula:

```text
mean = sum(values) / count
```

### Standard Deviation

The standard deviation measures how spread out the values are around the mean.

Formula:

```text
std = sqrt(sum((value - mean)^2) / count)
```

### Minimum

The smallest value of the feature.

### Percentiles

The program computes the following percentiles:

* 25%
* 50%
* 75%

The 50% percentile is also known as the median.

### Maximum

The largest value of the feature.

## Constraints

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

## Implementation Notes

The program follows these main steps:

1. Read the CSV file passed as argument.
2. Parse the header and rows.
3. Detect numerical columns.
4. Ignore non-numerical values and missing values.
5. Compute each statistic manually.
6. Display the result in a formatted table.

## Error Handling

The program should handle common errors such as:

* Missing argument
* Invalid file path
* Empty dataset
* Dataset with no numerical features
* Missing or invalid values inside numerical columns

## Example

```bash
./describe.py dataset_train.csv
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

## Goal

This part helps build a better intuition about the dataset before implementing logistic regression.

Instead of relying on existing tools, the purpose is to understand how descriptive statistics are computed internally and how to manipulate raw data manually.
