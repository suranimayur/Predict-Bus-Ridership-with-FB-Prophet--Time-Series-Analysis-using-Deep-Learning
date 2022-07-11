# Predict-Bus-Ridership-with-FB-Prophet--Time-Series-Analysis-using-Deep-Learning
Predict Bus Ridership with FB Prophet- Time Series Analysis using Deep Learning

Hellow Friend , Today we will make time series analysis with python project for Bus Ridership with Facebook Prophet.

Step-01 Import necessary modules and library for time series analysis in Python.

import numpy as np
import pandas as pd
from itertools import product
from fbprophet import Prophet
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from fbprophet.plot import plot_cross_validation_metric
from fbprophet.diagnostics import cross_validation, performance_metrics
