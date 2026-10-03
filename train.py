import os

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("iris_classification_experiment")


def main():
    n_estimators = 150
    max_depth = 15
    random_state = 42

    with mlflow.start_run(run_name="RandomForest_Run"):
        df = pd.read_csv("data/dataset.csv")
        X = df.drop(columns="target").values
        y = df["target"].values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("dataset_version", "v2.0")

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average="macro")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        for epoch in range(1, 6):
            mlflow.log_metric("train_loss", 0.5 / epoch, step=epoch)

        print(f"Эксперимент завершен. Accuracy: {acc:.4f}, F1: {f1:.4f}")

        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(5, 5))
        plt.title("Feature Importances")
        plt.bar(range(X.shape[1]), model.feature_importances_)
        plt.savefig("plots/feature_importance.png")
        plt.close()
        mlflow.log_artifact("plots/feature_importance.png", artifact_path="plots")

        mlflow.sklearn.log_model(model, artifact_path="random_forest_model")


if __name__ == "__main__":
    main()