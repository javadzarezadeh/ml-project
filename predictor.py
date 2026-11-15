import argparse
import ast
import os
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm



def main(features_path: str, targets_path: str, target: str):
    print("Loading datasets...")
    X = pd.read_pickle(features_path)
    Y = pd.read_pickle(targets_path)

    # Do train-test split, 80:20 ratio
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    # Define model, parameter choices are a bit arbitrary...
    print("Training LightGBM model...")
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=100,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    # Train model
    print("Training final regressor model with early stopping...")
    model.fit(
        X_fit, y_fit[target],
        eval_X = X_val, eval_y= y_val[target],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(stopping_rounds=70, verbose=False)]
    )

    # Collect and print test results
    preds = model.predict(X_test)
    r2 = r2_score(y_test[target], preds)
    rmse = np.sqrt(mean_squared_error(y_test[target], preds))
    mae = mean_absolute_error(y_test[target], preds)
    print(f"R2 Score: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicts ")
    parser.add_argument("--X", required=True)
    parser.add_argument("--Y", required=True)
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    main(args.X, args.Y, args.target)