import logging
import math

from scipy import signal
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def split_pids(current_pids_list, seed_value, split_ratio):
    """
    Function to split the pids into two arrays
    """
    list_pids = list(current_pids_list)
    list_pids.sort()
    logger.info("Current pids are: {}".format(str(list_pids)))
    person_count_test = math.floor(split_ratio * len(list_pids))
    person_count_train = len(list_pids) - person_count_test
    logger.info("Total persons in training: {}, testing/validation: {}".format(person_count_train, person_count_test))
    seed_value = int(seed_value)
    np.random.seed(seed_value)
    train_pids = np.random.choice(list_pids, person_count_train, replace=False)
    test_pids = np.array(list(set(list_pids) - set(train_pids)))
    return train_pids, test_pids


def apply_butter_worth_filter(ss, fc=20, sampling_freq=51.2, order=8):
    w = fc / (sampling_freq / 2)
    b, a = signal.butter(order, w, 'low')
    output = signal.filtfilt(b, a, ss)
    return output


def resample_signal(input_signal, resampled_length=250):
    curr_length = len(input_signal)
    idx = np.array(range(curr_length))
    idx_new = np.linspace(0, idx.max(), resampled_length)
    # linear interpolation
    f = interp1d(idx, input_signal, kind='cubic')
    resampled_signal = f(idx_new)
    return resampled_signal
