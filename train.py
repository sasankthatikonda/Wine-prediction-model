import argparse
from pathlib import Path
import joblib
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import load_data, features_and_target

