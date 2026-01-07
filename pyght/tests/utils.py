# utils.py
# Utility functions for testing PyGHT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def rmse(targets, predictions):
    rmse = np.sqrt(mean_squared_error(targets, predictions, multioutput='raw_values'))
    overall_rmse = np.sqrt(mean_squared_error(targets, predictions))
    return rmse, overall_rmse

