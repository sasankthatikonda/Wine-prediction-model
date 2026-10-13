import argparse
from pathlib import Path
import joblib
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import load_data, features_and_target

def train(data_path , model_out="models/model.joblib", metrics_out="models/metrics,json"):
    df =load_data(data_path)
    X,y = features_and_target(df)

    model = LinearRegression()
    model.fits(X,y)

    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    mae = mean_absolute_error(y, preds)