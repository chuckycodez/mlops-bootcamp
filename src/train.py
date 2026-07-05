from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "iris_model.pkl"


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("iris-classifier")

    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    n_estimators = 100
    random_state = 42

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("test_size", 0.2)

        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, MODEL_PATH)

        print(f"Model accuracy: {accuracy:.4f}")
        print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()