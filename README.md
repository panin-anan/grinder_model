# Grinder Model

## Overview

This repository provides a script for modeling a material removal predictive model during grinding operations using machine learning techniques. The code uses preprocessed grind data to train and evaluate machine learning models, enabling predictions of material removal (in volume) based on key grinding parameters: grind time, RPM, force, and belt wear which can be adjusted in main().

### Key Functions
- **Model Selection:** Mainly use Support Vector Machine (SVR). Other models are explored in the `volume-model-explore` branch.
- **Grid Search:** Automatically optimizes hyperparameters tuning for training models using GridSearchCV based on input criteria such as Mean Absolute Error (MAE).
- **Evaluation Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² scores.
- **Visualization:** Plots actual vs predicted values to better understand model's performance.

### Branch Summary

- `main`: contain scripts used for modeling and evaluating stationary grinding process.
- `moving_grinder`: modified scripts to use stationary grind predictive model to predict result from moving grinder on a robot tests (introduce feed_rate, num_pass, etc.)
- `feedrate_volume_model`: modified scripts for modeling and evaluating moving grind process.
- `volume_model_explore`: similar to main but contain functions for machine learning methods other than Support Vector Regression(SVR)

---
## Installation

### Dependencies
This project requires the following dependencies:
- Python 3.8+
- NumPy 1.24.0
- Pandas 2.2.3
- Scikit-learn 1.5.2
- Matplotlib 3.5.1
- Joblib 1.4.2

#### Installation via pip
```bash
pip install numpy==1.24.0 pandas==2.2.3 scikit-learn==1.5.2 matplotlib==3.5.1 joblib==1.4.2
```

#### Running the main model training script
The main script which performs loading grind data, training, and evaluating the model is `volume_model_svr.py`
path to run the code can change depending on how your workspace is setup:
```bash
python3 src/grinder_model/grinder_model/volume_model_svr.py 
```
---

## Data Preprocessing

The grind data is preprocessed using the `DataManager` class, which:
1. Loads test data in .csv format
2. Filters out noisy, irrelevant data points, or points with failure messages (e.g., high MAD RPM, duplicates, failures).
Filter can be adjusted in `data_manager.py`, specifically in `filter_grind_data` function.
---

## Script list and their usage

### volume_model_svr.py
Main script for training and evaluating predictive model.
Typically, the grind data in .csv format should contain the following columns:
- `grind_time`
- `avg_rpm`
- `avg_force`
- `grind_area`
- `initial_wear`
- `removed_material`

User can change up input and output variables to study different relations by modifying `related_columns` and `target_columns` variables in `main()` of the script.
After closing the evaluation graph, the script automatically save the trained model in the `saved_model` folder with the name which the user need to input as an argument for save_model() function. 
User will be asked whether they want to overwrite the model or not if the same name is used.

### volume_predictor_svr.py
A supplementary script which the author made used for evaluating/trying out the saved predictive model.
The code select a saved predictive model from `saved_model` folder according to the path which need to be input by the user in `main()`. 
Contain two usages that user will be prompted to select:
1. Use the model to predict `removed_material` through manual input of grinding parameter settings.
2. Load .csv data to serve as test data to evaluate the model. ( no training, only test)


### grind_settings_generator.py
With input of desired volume removal, automatically generate grinding parameter settings to achieve the desired removal through the use of a selected trained model by the user.
Model and scaler file path must be inputted in `main()`

Meant to be integrated in predictive grinding. (Work in Progress)

### model_sensitivity_analysis.py
Perform sensitivity analysis on each individual grind setting parameters in order to assess how the model perceive their relations with grinding removed volume.
A set of base values and variation ranges for the grinding parameter variables are set manually in `main`.


### rpm_correction_model.py
ONLY USE THIS WHEN RPM_SETPOINT AND ACTUAL RPM DIFFERS A LOT
Script for training model to compensate for when flow rate is limited.

---

## Existing model and examples

### Stationary grind model training
Before starting the script, input desired model and scaler file name in `main()` for saving the model in the `saved_model` folder
Start the script by running
```bash
python3 src/grinder_model/grinder_model/volume_model_svr.py 
```
The program will endlessly prompt for user to select test data in .csv file format to input.
After the user becomes content with the amount of data file selected, just press cancel for the program to continue the next step.
![image](https://github.com/user-attachments/assets/569c7767-62b0-4a97-9254-b7c3742823ec)

The program will then use the selected data to train and test the model based on SVR method, outputting a model performance graph shown below:

![image](https://github.com/user-attachments/assets/c933ebe2-62ba-4705-b9e2-c163fad44d41)


---
### Generate Grind setting
First set file path for model and scaler you wish to use, then input all the necessary constants for grinding parameters (such as rpm and belt width)
Note: with current iteration, RPM must be set by user to narrow down range of operation for grind settings.
Start the script by running

```bash
python3 src/grinder_model/grinder_model/grind_settings_generator.py 
```

The settings to achieve the input desired removal material volume should print out on the terminal:

![image](https://github.com/user-attachments/assets/3b87fcc8-0700-4f53-a612-d7916e4c39e9)



---
### Sensitivity Analysis
First set file path for model and scaler, input grind settings range the user wish to study and run the script:
```bash
python3 src/grinder_model/grinder_model/model_sensitivity_analysis.py 
```
`model_sensitivity_analysis.py` script output from existing saved model
![image](https://github.com/user-attachments/assets/719e9046-9979-48d1-b76d-33503f387e3b)



