"""
This example demonstrates how to specify pip requirements using `pip_requirements` and
`extra_pip_requirements` when logging a model via `mlflow.*.log_model`.

Uses GradientBoostingRegressor on California Housing instead of XGBoost on Iris.
"""

import os

import sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
from mlflow.artifacts import download_artifacts
from mlflow.models.signature import infer_signature


# Fix for paths with spaces
mlflow.set_tracking_uri("sqlite:///mlflow.db")


def read_lines(path):
    with open(path) as f:
        return f.read().splitlines()


def get_pip_requirements(run_id, artifact_path, return_constraints=False):
    req_path = download_artifacts(
        run_id=run_id, artifact_path=f"{artifact_path}/requirements.txt"
    )
    reqs = read_lines(req_path)

    if return_constraints:
        con_path = download_artifacts(
            run_id=run_id, artifact_path=f"{artifact_path}/constraints.txt"
        )
        cons = read_lines(con_path)
        return set(reqs), set(cons)

    return set(reqs)


def main():
    # Load California Housing dataset
    housing = fetch_california_housing()
    X_train, X_test, y_train, y_test = train_test_split(
        housing.data, housing.target, test_size=0.25, random_state=42
    )

    # Train a Gradient Boosting model
    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    signature = infer_signature(X_test, predictions)

    sklearn_req = f"scikit-learn=={sklearn.__version__}"

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        # Default (both `pip_requirements` and `extra_pip_requirements` are unspecified)
        artifact_path = "default"
        mlflow.sklearn.log_model(model, artifact_path, signature=signature)
        pip_reqs = get_pip_requirements(run_id, artifact_path)
        assert sklearn_req in pip_reqs, pip_reqs

        # Overwrite the default set of pip requirements using `pip_requirements`
        artifact_path = "pip_requirements"
        mlflow.sklearn.log_model(
            model, artifact_path, pip_requirements=[sklearn_req], signature=signature
        )
        pip_reqs = get_pip_requirements(run_id, artifact_path)
        assert sklearn_req in pip_reqs, pip_reqs

        # Add extra pip requirements on top of the default set
        artifact_path = "extra_pip_requirements"
        mlflow.sklearn.log_model(
            model, artifact_path, extra_pip_requirements=["pandas"], signature=signature
        )
        pip_reqs = get_pip_requirements(run_id, artifact_path)
        assert sklearn_req in pip_reqs, pip_reqs

        # Specify pip requirements using a requirements file
        req_file = os.path.join(os.getcwd(), "temp_requirements.txt")
        with open(req_file, "w") as f:
            f.write(sklearn_req)

        artifact_path = "requirements_file_path"
        mlflow.sklearn.log_model(
            model, artifact_path, pip_requirements=req_file, signature=signature
        )
        pip_reqs = get_pip_requirements(run_id, artifact_path)
        assert sklearn_req in pip_reqs, pip_reqs

        # List of pip requirement strings
        artifact_path = "requirements_file_list"
        mlflow.sklearn.log_model(
            model,
            artifact_path,
            pip_requirements=[sklearn_req, f"-r {req_file}"],
            signature=signature,
        )
        pip_reqs = get_pip_requirements(run_id, artifact_path)
        assert sklearn_req in pip_reqs, pip_reqs

        # Using a constraints file
        con_file = os.path.join(os.getcwd(), "temp_constraints.txt")
        with open(con_file, "w") as f:
            f.write(sklearn_req)

        artifact_path = "constraints_file"
        mlflow.sklearn.log_model(
            model,
            artifact_path,
            pip_requirements=[sklearn_req, f"-c {con_file}"],
            signature=signature,
        )
        pip_reqs, pip_cons = get_pip_requirements(
            run_id, artifact_path, return_constraints=True
        )
        assert sklearn_req in pip_reqs or "-c constraints.txt" in pip_reqs, pip_reqs
        assert pip_cons == {sklearn_req}, pip_cons

        # Cleanup temp files
        os.remove(req_file)
        os.remove(con_file)


if __name__ == "__main__":
    main()