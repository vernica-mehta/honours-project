#!/bin/bash

nohup python3 cannon-train.py sfh_100_10_20251102_130203 --kfold 10 --snr 500 > output.log 2>&1 &&
nohup python3 cannon-train.py sfh_100_10_20251102_130203 --kfold 10 --snr 250 > output.log 2>&1 &&
nohup python3 cannon-train.py sfh_100_10_20251102_130203 --kfold 10 --snr 100 > output.log 2>&1 &&
nohup python3 cannon-train.py sfh_100_10_20251102_130203 --kfold 10 --snr 50 > output.log 2>&1 &&
nohup python3 cannon-train.py sfh_100_10_20251102_130203 --kfold 10 --snr 25 > output.log 2>&1 &&
nohup python3 cannon-train.py sfh_100_10_20251102_130203 --kfold 10 --snr 10 > output.log 2>&1 &