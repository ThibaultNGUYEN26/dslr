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


def ft_std(values):
    count = ft_count(values)
    if count == 0:
        return None

    mean = ft_mean(values)
    variance = 0.0
    for value in values:
        variance += (value - mean) ** 2
    return sqrt(variance / count)


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


def describe_values(values):
    return {
        "Count": float(ft_count(values)),
        "Mean": ft_mean(values),
        "Std": ft_std(values),
        "Min": ft_min(values),
        "25%": ft_percentile(values, 25),
        "50%": ft_percentile(values, 50),
        "75%": ft_percentile(values, 75),
        "Max": ft_max(values),
    }
