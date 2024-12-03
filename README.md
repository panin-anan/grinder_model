# Grinder Model

## Overview

This repository provides a script for modeling material removal during grinding operations using machine learning techniques. The code uses preprocessed grind data to train and evaluate machine learning models, enabling predictions of material removal based on key parameters: grind time, RPM, force, and belt wear which can be adjusted in main().

### Key Functions
- **Model Selection:** Mainly use Support Vector Machine (SVR). Other models are explored in the volume-model-explore branch.
- **Grid Search:** Optimizes hyperparameters for each model using GridSearchCV.
- **Evaluation Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² scores.
- **Visualization:** Plots actual vs predicted values to better understand model's performance.
---
## Installation

### Dependencies
This project requires the following dependencies:
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

#### Installation via pip
```bash
pip install numpy pandas scikit-learn matplotlib joblib
```

#### Running the main script
The main script which performs loading grind data, training, and evaluating the model is `volume_model_svr.py`
path to run the code can change depending on how your workspace is setup:
```bash
python3 src/grinder_model/grinder_model/volume_model_svr.py 
```
---

## Data Preprocessing

The grind data is preprocessed using the `DataManager` class, which:
1. Loads test data in .csv format
2. Filters out noisy or irrelevant data points (e.g., high MAD RPM, duplicates, failures).

---

## Usage

### Data Input Format
Typically, the grind data must contain the following columns:
- `grind_time`
- `avg_rpm`
- `avg_force`
- `grind_area`
- `initial_wear`
- `removed_material`

User can change up input and output variables to study different relations by modifying `related_columns` and `target_columns` variables in main.

