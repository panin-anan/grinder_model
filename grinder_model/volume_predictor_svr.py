import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import joblib
import pathlib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

def open_file_dialog():
    # Create a Tkinter window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    # Open file dialog and return selected file path
    file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
    return file_path

def load_data(file_path):
    #Load dataset from a CSV file.
    return pd.read_csv(file_path)

def load_model(use_fixed_path=False, fixed_path='saved_models/svr_model.pkl'):
    if use_fixed_path:
        # If the argument is True, use the fixed path
        filepath = os.path.abspath(fixed_path)
        print(f"Using fixed path: {filepath}")
    else:
        # Open file dialog to manually select the model file
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Open the file dialog and allow the user to select the model file
        filepath = filedialog.askopenfilename(title="Select Model File", filetypes=[("Pickle files", "*.pkl")])
        
        if not filepath:
            print("No file selected. Exiting.")
            return None

    # Load the model
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model



def load_scaler(use_fixed_path=False, fixed_path='saved_models/scaler.pkl'):
    if use_fixed_path:
        # If the argument is True, use the fixed path
        filepath = os.path.abspath(fixed_path)
        print(f"Using fixed path for scaler: {filepath}")
    else:
        # Open file dialog to manually select the scaler file
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Open the file dialog and allow the user to select the scaler file
        filepath = filedialog.askopenfilename(title="Select Scaler File", filetypes=[("Pickle files", "*.pkl")])
        
        if not filepath:
            print("No file selected. Exiting.")
            return None

    # Load the scaler
    scaler = joblib.load(filepath)
    print(f"Scaler loaded from {filepath}")
    return scaler

def preprocess_test_data(data, target_column, scaler):
    #Preprocess the data by splitting into features and target and then scaling.
    X_test = data.drop(columns=target_column)
    y_test = data[target_column]

    # Feature scaling
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_test_scaled, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    # Evaluate the model with Mean Squared Error and R^2 Score
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) 
    mean_abs = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {mean_abs}")
    print(f"RMS Error: {rmse}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Plot actual vs predicted for each output
    plt.figure(figsize=(12, 6))
    
    for i, col in enumerate(y_test.columns):
        plt.subplot(1, len(y_test.columns), i + 1)
        plt.scatter(y_test[col], y_pred[:, i])
        
        # Set axis limits to be the same
        min_val = min(min(y_test[col]), min(y_pred[:, i]))
        max_val = max(max(y_test[col]), max(y_pred[:, i]))
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        
        # Plot reference diagonal line for perfect prediction
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')

        plt.xlabel(f"Actual {col}")
        plt.ylabel(f"Predicted {col}")
        plt.title(f"Actual vs Predicted {col}")
    
    plt.tight_layout()
    plt.show()


def main():
    #get grind model
    use_fixed_model_path = True# Set this to True or False based on your need
    
    if use_fixed_model_path:
        # Specify the fixed model and scaler paths
        fixed_model_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_model_svr_V2_angle.pkl'
        fixed_scaler_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_scaler_svr_V2_angle.pkl'
        
        grind_model = load_model(use_fixed_path=True, fixed_path=fixed_model_path)
        scaler = load_scaler(use_fixed_path=True, fixed_path=fixed_scaler_path)
    else:
        # Load model and scaler using file dialogs
        grind_model = load_model(use_fixed_path=False)
        scaler = load_scaler(use_fixed_path=False)

    '''
    #read current belt's 'initial wear', 'removed_volume', 'RPM' and predict 'Force' and 'grind_time'
    rpm_range = np.arange(8500, 9000, 100)  # from 8500 to 10000 in steps of 500
    force_range = np.arange(7, 9.1, 1)  # from 3 to 9 in steps of 1
    time_range = np.arange(12.5, 13.5, 0.5)
    initial_wear = 10000000

    for avg_rpm in rpm_range:
       for avg_force in force_range:
           for grind_time in time_range:
                # Create a DataFrame to store the input data
                input_data_dict = {
                    'grind_time': [grind_time],
                    'avg_rpm': [avg_rpm],
                    'avg_force': [avg_force],
                    'initial_wear': [initial_wear]
                }
                input_df = pd.DataFrame(input_data_dict)
                input_scaled = scaler.transform(input_df)
                input_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)
                # Predict volume
                predicted_volume = grind_model.predict(input_scaled)
                print(f"RPM: {avg_rpm}, Force: {avg_force}N, Grind Time: {grind_time} sec --> Predicted Removed Volume: {predicted_volume[0]}")
    '''
    
    #load test data and evaluate model
    #read grind data
    file_path = open_file_dialog()
    if not file_path:
        print("No file selected. Exiting.")
        return

    grind_data = load_data(file_path)


    # Delete rows where removed_material is less than 12
    grind_data = grind_data[grind_data['removed_material'] >= 5]

    # Filter out points which have mad of more than 1000
    grind_data = grind_data[grind_data['mad_rpm'] <= 1000]

    # Filter out avg rpm that is lower than half of rpm_setpoint
    grind_data = grind_data[grind_data['avg_rpm'] >= grind_data['rpm_setpoint'] / 2]

    grind_data = grind_data[pd.isna(grind_data['failure_msg'])]
    print(grind_data)

    #drop unrelated columns
    related_columns = [ 'grind_time', 'avg_rpm', 'avg_force', 'avg_pressure', 'initial_wear', 'removed_material']
    grind_data = grind_data[related_columns]

    #desired output
    target_columns = ['removed_material']

    # Preprocess the data (train the model using the CSV data, for example)
    X_test_scaled, y_test = preprocess_test_data(grind_data, target_columns, scaler)

    evaluate_model(grind_model, X_test_scaled, y_test)


if __name__ == "__main__":
    main()