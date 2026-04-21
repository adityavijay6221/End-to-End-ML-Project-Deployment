# MLOps Project

## Project Overview

A modular MLOps pipeline for predicting student performance, built with scikit-learn and Flask, structured for production readiness.

## Current Progress

### Completed
- [x] Project setup (`setup.py`, `requirements.txt`, virtual environment)
- [x] Custom logging (`src/logger.py`) — writes timestamped `.log` files to `logs/`
- [x] Custom exception handling (`src/exception.py`) — captures file name and line number
- [x] Utility functions (`src/utils.py`) — `save_object()` serializes objects to pickle files
- [x] Data ingestion (`src/components/data_ingestion.py`) — loads CSV, splits into train/test, saves to `artifacts/`
- [x] Data transformation (`src/components/data_transformation.py`) — preprocesses features, saves `preprocessor.pkl` (fitted) and transformed CSVs to `artifacts/`
- [x] Model training (`src/components/model_trainer.py`) — trains LinearRegression, Ridge, Lasso with GridSearchCV; saves best model to `artifacts/model.pkl`
- [x] Prediction pipeline (`src/pipeline/predict_pipeline.py`) — loads fitted preprocessor + model, exposes `PredictPipeline` and `CustomData`
- [x] Flask web app (`app.py`) — form-based UI at `/predict`, runs on port 8080

### In Progress / Not Started
- [ ] Model evaluation component
- [ ] Deployment / containerisation

## Project Structure

```
MLOps/
├── data/
│   ├── Students_Performance.csv    ← raw source data (input)
│   ├── EDA.ipynb                   ← exploratory data analysis
│   └── Model_training.ipynb        ← model training experiments
├── artifacts/
│   ├── data.csv                    ← raw data copy
│   ├── train.csv                   ← 80% training split
│   ├── test.csv                    ← 20% test split
│   ├── transformed_train.csv       ← preprocessed training data (no headers, last col = target)
│   ├── transformed_test.csv        ← preprocessed test data (no headers, last col = target)
│   ├── preprocessor.pkl            ← fitted ColumnTransformer (StandardScaler + OneHotEncoder)
│   └── model.pkl                   ← best trained model (Linear Regression / Ridge / Lasso)
├── src/
│   ├── logger.py                   ← logging setup
│   ├── exception.py                ← custom exception handler
│   ├── utils.py                    ← save_object() helper
│   ├── components/
│   │   ├── data_ingestion.py       ← data ingestion pipeline
│   │   ├── data_transformation.py  ← feature preprocessing pipeline
│   │   └── model_trainer.py        ← hyperparameter tuning + model selection
│   └── pipeline/
│       └── predict_pipeline.py     ← PredictPipeline + CustomData for inference
├── templates/
│   ├── index.html                  ← landing page
│   └── home.html                   ← prediction form + result
├── app.py                          ← Flask app (port 8080)
├── logs/                           ← auto-generated log files
├── setup.py
└── requirements.txt
```

## Data Flow

```
data/Students_Performance.csv
        ↓
artifacts/data.csv                  (raw copy)
artifacts/train.csv                 (80% split)
artifacts/test.csv                  (20% test split)
        ↓
artifacts/transformed_train.csv     (scaled + encoded, numpy array, no headers)
artifacts/transformed_test.csv      (scaled + encoded, numpy array, no headers)
artifacts/preprocessor.pkl          (fitted ColumnTransformer)
        ↓
artifacts/model.pkl                 (best model selected by R2)
```

## Feature Engineering (DataTransformation)

- **Target:** `Average Score`
- **Numerical features:** `math score`, `reading score`, `writing score`, `Total Score` → `StandardScaler`
- **Categorical features:** `gender`, `race/ethnicity`, `parental level of education`, `lunch`, `test preparation course` → `OneHotEncoder(handle_unknown='ignore')`
- Preprocessor is fit on train set only, then applied to test set
- `Total Score` is computed as `math + reading + writing` at inference time in `CustomData.as_dataframe()`

## Model Training (ModelTrainer)

- Models: `LinearRegression`, `Ridge`, `Lasso`
- Hyperparameter tuning: `GridSearchCV(cv=3)` over `alpha: [0.1, 1.0, 10.0]` for Ridge and Lasso
- Best model selected by R2 score on test set, saved to `artifacts/model.pkl`

## Dependencies

- `pandas`, `numpy` — data manipulation
- `scikit-learn` — ML models, preprocessing, train/test split
- `flask` — web app
- `matplotlib`, `seaborn` — visualization

## Running the Project

Always run from the project root (`MLOps/`):

```bash
# Step 1 — ingest data
python3 src/components/data_ingestion.py

# Step 2 — transform features and save fitted preprocessor
python3 src/components/data_transformation.py

# Step 3 — train and select best model
python3 src/components/model_trainer.py

# Step 4 — start the web app
python3 app.py
# Open http://localhost:8080
```
