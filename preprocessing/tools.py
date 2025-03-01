import numpy as np

def calculate_displacement_difference(y_i, y_j):
    return np.abs(y_i - y_j).sum()


def calculate_velocity_difference(y_i, y_j):
    return np.abs(np.diff(y_i) - np.diff(y_j)).sum()


def calculate_acceleration_difference(y_i, y_j):
    return np.abs(np.diff(y_i, n=2) - np.diff(y_j, n=2)).sum()


def calculate_gray_b_correlation(y_i, y_j):
    n = len(y_i)
    displacement_diff = calculate_displacement_difference(y_i, y_j)
    velocity_diff = calculate_velocity_difference(y_i, y_j)
    acceleration_diff = calculate_acceleration_difference(y_i, y_j)
    return 1 / (1 + (displacement_diff / n) + (velocity_diff / (n-1)) + (acceleration_diff / (n-2)))


def calculate_total_correlation_energy(signals):
    n_signals = len(signals)
    total_energy = np.zeros(n_signals)
    for i in range(n_signals):
        for j in range(n_signals):
            if i != j:
                correlation_energy = calculate_gray_b_correlation(signals[i], signals[j])
                total_energy[i] += correlation_energy ** 2
    return total_energy


def calculate_weights(total_energy):
    weights = total_energy / total_energy.sum()
    return weights


def calculate_fused_signal(signals, weights):
    fused_signal = np.zeros_like(signals[0]).astype("float64")
    for i in range(len(signals)):
        fused_signal += weights[i] * signals[i]
    return fused_signal

from statsmodels.tsa.ar_model import AutoReg
import scipy

def CalculateStatisticalFeatuers(raw, window_size=2000):
    res = None
    for i in range(0, len(raw) - window_size + 1, window_size):
        item = raw[i:i + window_size]

        # initial
        # add mean
        item_mean = item.mean(axis=0).reshape(1, -1)
        each_row = item_mean

        # add std
        item_std = item.std(axis=0).reshape(1, -1)
        each_row = np.hstack((each_row, item_std))

        # add median absolute deviation
        item_mad = scipy.stats.median_abs_deviation(item, axis=0).reshape(1, -1)
        each_row = np.hstack((each_row, item_mad))

        # add max
        item_max = item.max(axis=0).reshape(1, -1)
        each_row = np.hstack((each_row, item_max))

        # add min
        item_min = item.min(axis=0).reshape(1, -1)
        each_row = np.hstack((each_row, item_min))

        # add energy. Energy measure. Sum of the squares divided by the number of values.
        item_energy = np.mean((item - item_mean.flatten()) ** 2, axis=0).reshape(1, -1)
        each_row = np.hstack((each_row, item_energy))

        # add iqr. Interquartile range
        q1 = np.percentile(item, 25, axis=0).reshape(1, -1)  # Calculate the first quartile (Q1)
        q3 = np.percentile(item, 75, axis=0).reshape(1, -1)  # Calculate the third quartile (Q3)
        item_iqr = q3 - q1
        each_row = np.hstack((each_row, item_iqr))

        # add entropy: Signal entropy
        item_entropy = scipy.stats.entropy(item+1e-7, base=2).reshape(1, -1)
        item_entropy = np.clip(item_entropy, 0, 10).astype(np.float32)
        each_row = np.hstack((each_row, item_entropy))

        # add arCoeff: Autorregresion coefficients with Burg order equal to 4
        item_arCoeff = AutoReg(item, lags=1).fit().params[1:].reshape(1, -1)
        each_row = np.hstack((each_row, item_arCoeff))

        if res is None:
            res = each_row
        else:
            res = np.vstack((res, each_row))
    return res