# MLflow Experiment Tracking — California Housing Price Prediction

This project demonstrates **MLflow Experiment Tracking** using the California Housing dataset with a Gradient Boosting Regressor. The pipeline trains multiple models with different hyperparameter configurations, logs everything to MLflow, and enables side-by-side comparison through the MLflow UI.

## Objective

- Log model parameters and metrics across multiple training runs
- Compare model performance with different hyperparameter configurations
- Store and load trained models using MLflow's model registry
- Manage pip requirements and dependencies for model serving
- Demonstrate autologging vs manual logging approaches

## Dataset

The **California Housing dataset** is a built-in sklearn dataset containing 20,640 samples with 8 features:

- `MedInc` — Median income in block group
- `HouseAge` — Median house age in block group
- `AveRooms` — Average number of rooms per household
- `AveBedrms` — Average number of bedrooms per household
- `Population` — Block group population
- `AveOccup` — Average number of household members
- `Latitude` — Block group latitude
- `Longitude` — Block group longitude

**Target:** `MedHouseVal` — Median house value (in units of $100,000)

## Model

**GradientBoostingRegressor** from scikit-learn — a tree-based ensemble method that builds sequential decision trees, where each tree corrects the errors of the previous one. Hyperparameters tuned across experiments:

| Parameter | Description |
|-----------|-------------|
| `n_estimators` | Number of boosting stages (trees) |
| `max_depth` | Maximum depth of each individual tree |
| `learning_rate` | Shrinkage factor applied to each tree's contribution |

## Project Structure

```
├── housing_prediction.py      # Main training script with MLflow parameter/metric/model logging
├── serving.py                 # Pip requirements and dependency management for model serving
├── starter.ipynb              # Autologging vs manual logging walkthrough notebook
├── requirements.txt           # Python dependencies
└── README.md
```

### File Descriptions

**`housing_prediction.py`** — Trains a `GradientBoostingRegressor` on the California Housing dataset. Accepts command-line arguments for hyperparameters, logs `n_estimators`, `max_depth`, `learning_rate` as parameters, logs `RMSE`, `MAE`, `R2` as metrics, and stores the trained model with an inferred signature to MLflow.

**`serving.py`** — Demonstrates how to manage pip requirements when logging models with MLflow. Covers default requirements, custom `pip_requirements`, `extra_pip_requirements`, requirements files, and constraints files — essential for reproducible model deployment.

**`starter.ipynb`** — Walkthrough notebook covering:
- MLflow autologging (automatic parameter/metric capture)
- Manual logging with `mlflow.start_run()`
- Storing models with `mlflow.sklearn.log_model()`
- Loading saved models with `mlflow.sklearn.load_model()`

## Setup & Installation

```bash
git clone https://github.com/Chitra-Pahuja/mlflow-california-housing-experiment-tracking.git
cd mlflow-california-housing-experiment-tracking

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

## Running Experiments

Run the training script with different hyperparameter combinations:

```bash
# Run 1: Default (n_estimators=100, max_depth=3, learning_rate=0.1)
python housing_prediction.py

# Run 2: More trees, deeper, slower learning
python housing_prediction.py 200 5 0.05

# Run 3: Fewer trees, faster learning
python housing_prediction.py 50 3 0.2
```

Run the serving demo:
```bash
python serving.py
```

## Viewing Results in MLflow UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Open http://localhost:5001 to view and compare experiment runs.

## MLflow Dashboard

![MLflow Training Runs](MLflow%20dashboard.png)

## Experiment Results

| Run | n_estimators | max_depth | learning_rate | RMSE | MAE | R² |
|-----|-------------|-----------|---------------|------|-----|-----|
| 1 | 100 | 3 | 0.10 | 0.548 | 0.375 | 0.779 |
| 2 | 200 | 5 | 0.05 | 0.502 | 0.334 | 0.815 |
| 3 | 50 | 3 | 0.20 | 0.549 | 0.374 | 0.779 |

**Best configuration:** Run 2 (`n_estimators=200`, `max_depth=5`, `learning_rate=0.05`) achieved the lowest RMSE (0.502) and highest R² (0.815).

## Key Concepts Demonstrated

1. **Experiment Tracking** — Logging parameters, metrics, and models for each training run
2. **Run Comparison** — Using the MLflow UI to compare hyperparameter configurations side-by-side
3. **Model Logging** — Saving trained sklearn models with input/output signatures
4. **Model Registry** — Registering best models for versioning and deployment
5. **Autologging vs Manual Logging** — Comparing both approaches in the starter notebook
6. **Dependency Management** — Specifying pip requirements and constraints for reproducible serving

## Tech Stack

- **Python 3.10**
- **MLflow 3.10.1** — Experiment tracking, model logging, model registry
- **scikit-learn** — GradientBoostingRegressor, train/test split, evaluation metrics
- **pandas / numpy** — Data handling
