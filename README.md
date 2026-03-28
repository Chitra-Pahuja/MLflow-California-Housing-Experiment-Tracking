# MLflow Experiment Tracking — California Housing

MLflow experiment tracking lab using the California Housing dataset with Gradient Boosting Regressor. Based on the MLflow Lab1 structure from Northeastern University's MLOps course, with a different dataset, model, and hyperparameters.

## What's Different from the Base Lab

| Component | Base Lab | This Version |
|-----------|----------|-------------|
| Dataset | Wine Quality (UCI) | California Housing (sklearn) |
| Model | ElasticNet | GradientBoostingRegressor |
| Params | alpha, l1_ratio | n_estimators, max_depth, learning_rate |
| Target | Wine quality score (3–9) | Median house value ($100k) |

## Project Structure

```
├── housing_prediction.py      # Main training script with MLflow logging (mirrors linear_regression.py)
├── serving.py                 # Pip requirements demo for model serving
├── starter.ipynb              # Autologging vs manual logging walkthrough
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Run Experiments

```bash
# Default parameters (n_estimators=100, max_depth=3, learning_rate=0.1)
python housing_prediction.py

# Custom parameters
python housing_prediction.py 200 5 0.05
python housing_prediction.py 50 3 0.2

# Run serving demo
python serving.py
```

## View Results

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open http://localhost:5001 to compare experiment runs.

## MLflow Dashboard

Screenshot of the MLflow UI showing 3 training runs with different hyperparameters:

![MLflow Training Runs](mlflow_dashboard.png)

## Tech Stack

- **Python 3.10**
- **MLflow 3.10.1** — experiment tracking, model logging, serving
- **scikit-learn** — GradientBoostingRegressor
- **pandas / numpy** — data handling
