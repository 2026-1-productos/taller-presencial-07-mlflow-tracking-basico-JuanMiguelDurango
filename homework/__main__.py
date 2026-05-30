"""Entry point for the homework package."""

# homework/__main__.py
import argparse
import os

import mlflow
from sklearn.datasets import load_diabetes
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["elasticnet", "knn"])
    args = parser.parse_args()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("homework_experiment")

    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if args.model == "elasticnet":
        model = ElasticNet(random_state=42)
    else:
        model = KNeighborsRegressor()

    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        mlflow.log_param("model", args.model)
        mlflow.log_metric("mse", mse)

    print(f"Entrenamiento terminado con {args.model}")


if __name__ == "__main__":
    main()
