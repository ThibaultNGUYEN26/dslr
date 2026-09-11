from math import sqrt


def ft_count(values):
    total = 0
    for _ in values:
        total += 1
    return total


def ft_sum(values):
    total = 0.0
    for value in values:
        total += value
    return total


def ft_mean(values):
    count = ft_count(values)
    if count == 0:
        return None
    return ft_sum(values) / count


def ft_variance(values):
    count = ft_count(values)
    if count == 0:
        return None

    mean = ft_mean(values)
    variance = 0.0
    for value in values:
        variance += (value - mean) ** 2
    return variance / count


def ft_std(values):
    variance = ft_variance(values)
    if variance is None:
        return None
    return sqrt(variance)


def ft_min(values):
    smallest = None
    for value in values:
        if smallest is None or value < smallest:
            smallest = value
    return smallest


def ft_max(values):
    largest = None
    for value in values:
        if largest is None or value > largest:
            largest = value
    return largest


def ft_percentile(values, percentile):
    count = ft_count(values)
    if count == 0:
        return None
    if count == 1:
        return values[0]

    sorted_values = sorted(values)
    rank = (percentile / 100.0) * (count - 1)
    lower_index = int(rank)
    upper_index = lower_index + 1
    if upper_index >= count:
        return sorted_values[lower_index]

    fraction = rank - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * fraction


def describe_values(values, total_count=None):
    count = ft_count(values)
    if total_count is None:
        total_count = count

    minimum = ft_min(values)
    percentile_25 = ft_percentile(values, 25)
    percentile_75 = ft_percentile(values, 75)
    maximum = ft_max(values)

    return {
        "Count": float(count),
        "Missing": float(total_count - count),
        "Mean": ft_mean(values),
        "Std": ft_std(values),
        "Variance": ft_variance(values),
        "Min": minimum,
        "25%": percentile_25,
        "50%": ft_percentile(values, 50),
        "75%": percentile_75,
        "Max": maximum,
        "Range": maximum - minimum if minimum is not None else None,
        "IQR": (
            percentile_75 - percentile_25
            if percentile_25 is not None and percentile_75 is not None
            else None
        ),
    }
