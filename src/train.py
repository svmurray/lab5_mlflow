import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():
    mlflow.set_experiment("iris_model_exp")
    
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.25, random_state=42, stratify=data.target
    )

    C = 1.0
    max_iter = 200

    with mlflow.start_run(run_name="iris-logreg"):
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)

        clf = LogisticRegression(C=C, max_iter=max_iter, n_jobs=None)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_metric("accuracy", float(acc))
        mlflow.sklearn.log_model(clf, name="iris-classifier")

        print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()