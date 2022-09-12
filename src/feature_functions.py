import numpy as np
import pandas as pd
import math


def first_quartile(series):
    return np.quantile(series, 0.25)


def third_quartile(series):
    return np.quantile(series, 0.75)


def median(series):
    return np.quantile(series, 0.5)


def mean(series):
    return sum(series) / float(len(series))


def fft_coefficients(series: list, coefficients: int = 20) -> np.array:
    """Compute the Fourier transform coefficients of a series

    This method computes the first n coefficients of the Fourier transform of
    a  signal. The signal is padded with zeroes before the computation.

    Parameters
    ----------
    series: list
        The series to compute the coefficients from.
    coefficients: int
        The number of desired coeffcients.

    Returns
    -------
    np.array
        An array containing the Fourier coefficients.
    """
    s = pad_series_with_zeroes(series)
    t = np.fft.fft(s)

    components = [np.sqrt(i.real * i.real + i.imag * i.imag) for i in t]
    components.sort(reverse=True)

    if len(components) >= coefficients:
        return np.array(components[:coefficients])
    else:
        return np.array(components + [0] * (coefficients - len(components)))


def pad_series_with_zeroes(series: list) -> np.array:
    """Pad a signal with zeroes up to the closest power of 2

    This method extends a list with zeroes until its length is a power of 2.
    The method does not change the input list, but rather create a new list.

    Parameters
    ----------
    series: list
        The series to pad.

    Returns
    -------
    np.array
        The padded series, with the length equal to a power of 2.
    """
    if len(series) == closest_power_of_two(len(series)):
        return series.copy()

    c = closest_power_of_two(len(series))
    return np.array(list(series.copy()) + [0] * (c - len(series)))


def closest_power_of_two(number: int) -> int:
    """Find the closest power of 2

    This method returns the closest power of 2 to the passed number.

    Parameters
    ----------
    number: int
        The number to get the closest power of 2 from.

    Returns
    -------
    int
        The closest power of 2 to the passed number.
    """
    return 1 if number == 0 else 2 ** (number - 1).bit_length()


def normalize_vector(
        series: list, new_min: float = 0, new_max: float = 1) -> np.ndarray:
    """Normalize a vector

    This method scales a given vector between a new minimum and a new maximum.
    If the new range is not provided, the 0-1 range is used by default.

    Parameters
    ----------
    series : list
        The vector that should be normalized.
    new_min : double
        The new minimum to use in the vector normalization.
    new_max : double
        The new maximum to use in the vector normalization.

    Returns
    -------
    np.ndarray
        A normalized copy of the input vector.
    """
    upper = (series - np.min(series)) * (new_max - new_min)
    return upper / (np.max(series) - np.min(series)) + new_min


def full_range(series: list) -> float:
    """Compute the range of a series

    This method simply returns the range of a given series, computed as the
    maximum valune minus the minimum value.

    Parameters
    ----------
    series : list
        The series from which the range should be extracted.

    Returns
    -------
    float
        The range of the given series.
    """
    return series.max() - series.min()


def mean_positive(series) -> float:
    res = series[series > 0]
    if res.shape[0] == 0:
        return 0

    return 0 if math.isnan(float(res.mean())) else res.mean()


def mean_negative(series) -> float:
    res = series[series > 0]
    if res.shape[0] == 0:
        return 0

    return 0 if math.isnan(float(res.mean())) else res.mean()


def sad(series) -> float:
    return np.sum([np.abs(x[0] - x[1]) for x in zip(series[::], series[1::])])


def energy_entropy(series):
    sum_ = 0
    entropy = 0
    freq = 0

    for x in range(len(series)):
        value = series[x]
        sum_ += value * value

    for x in range(len(series)):
        value = series[x]
        freq = ((value * value) / sum_)
        if freq != 0:
            entropy += (freq * (np.log(freq) / np.log(2)))

    return 0 if math.isnan(float(-entropy)) else -entropy


def magnitude(series1, series2, series3):
    result_series = pd.Series()

    for x in range(len(series1)):
        acc_x = series1[x]
        acc_y = series2[x]
        acc_z = series3[x]
        result = np.sqrt(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z)
        result_series.at[x] = result

    return result_series


def pitch(series1, series2, series3):
    result_series = pd.Series()

    for x in range(len(series1)):
        gyro_x = series1[x]
        gyro_y = series2[x]
        gyro_z = series3[x]
        denominator = np.sqrt(gyro_y * gyro_y + gyro_z * gyro_z)

        result = np.arctan(gyro_x / denominator) * (180.0 / np.pi)
        result_series.at[x] = result

    return result_series


def roll(series1, series2, series3):
    result_series = pd.Series()

    for x in range(len(series1)):
        gyro_x = series1[x]
        gyro_y = series2[x]
        gyro_z = series3[x]
        denominator = np.sqrt(gyro_x * gyro_x + gyro_z * gyro_z)
        result = np.arctan(gyro_y / denominator) * (180.0 / np.pi)

        result_series.at[x] = result

    return result_series


def harmonic_ratio(series):
    sum_even = 0
    sum_ = 0

    for x in range(len(series)):
        val = series[x]
        sum_ += val

        if val % 2 == 0:
            sum_even += val

    return sum_even / sum_


def energy_average(series):
    sum_pos_energy = 0
    sum_neg_energy = 0

    for x in range(len(series)):
        temp_value = series[x] - series[0]
        if temp_value < 0:
            sum_neg_energy += temp_value * temp_value
        elif temp_value > 0:
            sum_pos_energy += temp_value * temp_value

    energy_average = (sum_pos_energy + sum_neg_energy) / len(series)

    return energy_average


def magnitude_single(x, y, z):
    acc_x = x
    acc_y = y
    acc_z = z

    result = np.sqrt(acc_x * acc_x + acc_y * acc_y + acc_z * acc_z)

    return result


def pitch_single(x, y, z):
    gyro_x = x
    gyro_y = y
    gyro_z = z

    denominator = np.sqrt(gyro_y * gyro_y + gyro_z * gyro_z)

    result = np.arctan(gyro_x / denominator) * (180.0 / np.pi)

    return result


def roll_single(x, y, z):
    gyro_x = x
    gyro_y = y
    gyro_z = z

    denominator = np.sqrt(gyro_x * gyro_x + gyro_z * gyro_z)
    result = np.arctan(gyro_y / denominator) * (180.0 / np.pi)

    return result


def sum_abs_diff(series):
    n = len(series)
    absolute_sum = []

    for i in range(0, n):
        total = 0
        for j in range(0, i):
            total += abs(series[i] - series[j])
        absolute_sum.append(total)

    return sum(absolute_sum)


if __name__ == "__main__":
    sample_series = [1, 2, 3, 4, 5]
    print(sum_abs_diff(sample_series))
    print(energy_average(sample_series))
    print(harmonic_ratio(sample_series))
    print(harmonic_ratio(sample_series))