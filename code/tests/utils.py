# utils.py
# Utility functions for testing PyGHT

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def rmse(targets, predictions, with_z=False):
    rmse = np.sqrt(mean_squared_error(targets, predictions, multioutput='raw_values'))
    if with_z:
        overall_sfh = np.sqrt(mean_squared_error(targets[:,:10], predictions[:,:10]))
        overall_z = np.sqrt(mean_squared_error(targets[:,10:], predictions[:,10:]))
        overall_rmse = (overall_sfh, overall_z)
    else:
        overall_rmse = np.sqrt(mean_squared_error(targets, predictions))
    return rmse, overall_rmse

def mae(targets, predictions, with_z=False):
    mae = mean_absolute_error(targets, predictions, multioutput='raw_values')
    if with_z:
        overall_sfh = mean_absolute_error(targets[:,:10], predictions[:,:10])
        overall_z = mean_absolute_error(targets[:,10:], predictions[:,10:])
        overall_mae = (overall_sfh, overall_z)
    else:
        overall_mae = mean_absolute_error(targets, predictions)
    return mae, overall_mae
