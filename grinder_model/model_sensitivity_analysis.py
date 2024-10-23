import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import joblib
import pathlib
import matplotlib.pyplot as plt

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

def sensitivity_analysis(model, scaler, base_data):
    """
    Perform sensitivity analysis on the trained model by varying grind time, RPM, force, and initial wear.
    Parameters:
        model: Trained SVR model.
        scaler: Scaler used to preprocess the input data.
        base_data: A single data point representing the manually input base values.
    """
    # Select the base data point for sensitivity analysis
    base_data = base_data.iloc[0]  # Use the first (and only) row for sensitivity analysis

    # Variables to test
    grind_times = np.linspace(base_data['grind_time'] * 0.2, base_data['grind_time'] * 2.5, 100)  # Vary grind time ±50%
    rpms = np.linspace(base_data['avg_rpm'] * 0.5, base_data['avg_rpm'] * 1.5, 100)  # Vary RPM ±50%
    forces = np.linspace(base_data['avg_force'] * 0.2, base_data['avg_force'] * 2, 100)  # Vary force ±50%
    wears = np.linspace(base_data['initial_wear'] * 0, base_data['initial_wear'] * 3.0, 100)  # Vary initial wear ±50%

   # Function to make predictions after scaling inputs
    def predict_volume(data_row):
        # Ensure the data_row is in the form of a DataFrame with proper column names
        if isinstance(data_row, pd.Series):
            data_row = pd.DataFrame([data_row])

        # Scale the input data using the scaler (it expects a DataFrame with feature names)
        scaled_row = scaler.transform(data_row)
        
        # Predict the volume using the trained model (which was also fitted with feature names)
        return model.predict(scaled_row)[0]

     # Create a single figure to plot all sensitivities together
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Sensitivity to grind time
    grind_time_sensitivities = []
    for grind_time in grind_times:
        test_data = base_data.copy()
        test_data['grind_time'] = grind_time
        predicted_volume = predict_volume(test_data)
        grind_time_sensitivities.append(predicted_volume)

    axs[0, 0].plot(grind_times, grind_time_sensitivities, color='blue')
    axs[0, 0].set_title('Grind Time Sensitivity')
    axs[0, 0].set_xlabel('Grind Time')
    axs[0, 0].set_ylabel('Predicted Volume')

    # Sensitivity to RPM
    rpm_sensitivities = []
    for rpm in rpms:
        test_data = base_data.copy()
        test_data['avg_rpm'] = rpm
        predicted_volume = predict_volume(test_data)
        rpm_sensitivities.append(predicted_volume)

    axs[0, 1].plot(rpms, rpm_sensitivities, color='green')
    axs[0, 1].set_title('RPM Sensitivity')
    axs[0, 1].set_xlabel('Average RPM')
    axs[0, 1].set_ylabel('Predicted Volume')

    # Sensitivity to force
    force_sensitivities = []
    for force in forces:
        test_data = base_data.copy()
        test_data['avg_force'] = force
        predicted_volume = predict_volume(test_data)
        force_sensitivities.append(predicted_volume)

    axs[1, 0].plot(forces, force_sensitivities, color='red')
    axs[1, 0].set_title('Force Sensitivity')
    axs[1, 0].set_xlabel('Average Force')
    axs[1, 0].set_ylabel('Predicted Volume')

    # Sensitivity to initial wear
    wear_sensitivities = []
    for wear in wears:
        test_data = base_data.copy()
        test_data['initial_wear'] = wear
        predicted_volume = predict_volume(test_data)
        wear_sensitivities.append(predicted_volume)

    axs[1, 1].plot(wears, wear_sensitivities, color='orange')
    axs[1, 1].set_title('Initial Wear Sensitivity')
    axs[1, 1].set_xlabel('Initial Wear')
    axs[1, 1].set_ylabel('Predicted Volume')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def sensitivity_analysis_rpm_forces(model, scaler, base_data, base_forces):
    """
    Perform sensitivity analysis on the trained model by varying grind time, RPM, force, and initial wear.
    Parameters:
        model: Trained SVR model.
        scaler: Scaler used to preprocess the input data.
        base_data: A single data point representing the manually input base values.
    """
    # Select the base data point for sensitivity analysis
    base_data = base_data.iloc[0]  # Use the first (and only) row for sensitivity analysis

    # Variables to test
    grind_times = np.linspace(base_data['grind_time'] * 0.2, base_data['grind_time'] * 2.5, 100)  # Vary grind time ±50%
    rpms = np.linspace(base_data['avg_rpm'] * 0.5, base_data['avg_rpm'] * 1.5, 100)  # Vary RPM ±50%
    forces = np.linspace(base_data['avg_force'] * 0.2, base_data['avg_force'] * 2, 100)  # Vary force ±50%
    wears = np.linspace(base_data['initial_wear'] * 0, base_data['initial_wear'] * 3.0, 100)  # Vary initial wear ±50%

    # Function to make predictions after scaling inputs
    def predict_volume(data_row):
        # Ensure the data_row is in the form of a DataFrame with proper column names
        if isinstance(data_row, pd.Series):
            data_row = pd.DataFrame([data_row])

        # Scale the input data using the scaler (it expects a DataFrame with feature names)
        scaled_row = scaler.transform(data_row)
        
        # Predict the volume using the trained model (which was also fitted with feature names)
        return model.predict(scaled_row)[0]

        # Plot sensitivity analysis results
    plt.figure(figsize=(10, 6))

    # Loop through each base force and calculate RPM sensitivities
    for base_force in base_forces:
        # Store results for RPM sensitivity
        rpm_sensitivities = []
        for rpm in rpms:
            test_data = base_data.copy()
            test_data['avg_rpm'] = rpm
            test_data['avg_force'] = base_force  # Set the base force for this iteration
            predicted_volume = predict_volume(test_data)
            rpm_sensitivities.append(predicted_volume)

        # Plot the RPM sensitivities for this base force
        plt.plot(rpms, rpm_sensitivities, label=f'Force: {base_force}N')

    # Customize the plot
    plt.xlabel('Average RPM')
    plt.ylabel('Predicted Grinded Volume')
    plt.title('Sensitivity to RPM for Different Base Forces')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def sensitivity_analysis_forces_rpms(model, scaler, base_data, base_rpms):
    """
    Perform sensitivity analysis on the trained model by varying grind time, RPM, force, and initial wear.
    Parameters:
        model: Trained SVR model.
        scaler: Scaler used to preprocess the input data.
        base_data: A single data point representing the manually input base values.
    """
    # Select the base data point for sensitivity analysis
    base_data = base_data.iloc[0]  # Use the first (and only) row for sensitivity analysis

    # Variables to test
    grind_times = np.linspace(base_data['grind_time'] * 0.2, base_data['grind_time'] * 2.5, 100)  # Vary grind time ±50%
    rpms = np.linspace(base_data['avg_rpm'] * 0.5, base_data['avg_rpm'] * 1.5, 100)  # Vary RPM ±50%
    forces = np.linspace(base_data['avg_force'] * 0.2, base_data['avg_force'] * 2, 100)  # Vary force ±50%
    wears = np.linspace(base_data['initial_wear'] * 0, base_data['initial_wear'] * 3.0, 100)  # Vary initial wear ±50%

    # Function to make predictions after scaling inputs
    def predict_volume(data_row):
        # Ensure the data_row is in the form of a DataFrame with proper column names
        if isinstance(data_row, pd.Series):
            data_row = pd.DataFrame([data_row])

        # Scale the input data using the scaler (it expects a DataFrame with feature names)
        scaled_row = scaler.transform(data_row)
        
        # Predict the volume using the trained model (which was also fitted with feature names)
        return model.predict(scaled_row)[0]

        # Plot sensitivity analysis results
    plt.figure(figsize=(10, 6))

    # Loop through each base force and calculate RPM sensitivities
    for base_rpm in base_rpms:
        # Store results for RPM sensitivity
        force_sensitivities = []
        for force in forces:
            test_data = base_data.copy()
            test_data['avg_rpm'] = base_rpm
            test_data['avg_force'] = force  # Set the base force for this iteration
            predicted_volume = predict_volume(test_data)
            force_sensitivities.append(predicted_volume)

        # Plot the RPM sensitivities for this base force
        plt.plot(forces, force_sensitivities, label=f'RPM: {base_rpm}')

    # Customize the plot
    plt.xlabel('Average Force')
    plt.ylabel('Predicted Grinded Volume')
    plt.title('Sensitivity to Forces for Different Base RPMs')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def sensitivity_analysis_time_forces(model, scaler, base_data, base_forces):
    """
    Perform sensitivity analysis on the trained model by varying grind time, RPM, force, and initial wear.
    Parameters:
        model: Trained SVR model.
        scaler: Scaler used to preprocess the input data.
        base_data: A single data point representing the manually input base values.
    """
    # Select the base data point for sensitivity analysis
    base_data = base_data.iloc[0]  # Use the first (and only) row for sensitivity analysis

    # Variables to test
    grind_times = np.linspace(base_data['grind_time'] * 0.2, base_data['grind_time'] * 2.5, 100)  # Vary grind time ±50%
    rpms = np.linspace(base_data['avg_rpm'] * 0.5, base_data['avg_rpm'] * 1.5, 100)  # Vary RPM ±50%
    forces = np.linspace(base_data['avg_force'] * 0.2, base_data['avg_force'] * 2, 100)  # Vary force ±50%
    wears = np.linspace(base_data['initial_wear'] * 0, base_data['initial_wear'] * 3.0, 100)  # Vary initial wear ±50%

    # Function to make predictions after scaling inputs
    def predict_volume(data_row):
        # Ensure the data_row is in the form of a DataFrame with proper column names
        if isinstance(data_row, pd.Series):
            data_row = pd.DataFrame([data_row])

        # Scale the input data using the scaler (it expects a DataFrame with feature names)
        scaled_row = scaler.transform(data_row)
        
        # Predict the volume using the trained model (which was also fitted with feature names)
        return model.predict(scaled_row)[0]

        # Plot sensitivity analysis results
    plt.figure(figsize=(10, 6))

    # Loop through each base force and calculate RPM sensitivities
    for base_force in base_forces:
        # Store results for RPM sensitivity
        time_sensitivities = []
        for time in grind_times:
            test_data = base_data.copy()
            test_data['grind_time'] = time
            test_data['avg_force'] = base_force  # Set the base force for this iteration
            predicted_volume = predict_volume(test_data)
            time_sensitivities.append(predicted_volume)

        # Plot the RPM sensitivities for this base force
        plt.plot(grind_times, time_sensitivities, label=f'Force: {base_force} N')

    # Customize the plot
    plt.xlabel('Grind Time')
    plt.ylabel('Predicted Grinded Volume')
    plt.title('Sensitivity to grind time for Different Base Forces')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    #get grind model
    use_fixed_model_path = True# Set this to True or False based on your need
    
    if use_fixed_model_path:
        # Specify the fixed model and scaler paths
        fixed_model_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_model_poly_V1.pkl'
        fixed_scaler_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_scaler_poly_V1.pkl'
        
        volume_model = load_model(use_fixed_path=True, fixed_path=fixed_model_path)
        scaler = load_scaler(use_fixed_path=True, fixed_path=fixed_scaler_path)
    else:
        # Load model and scaler using file dialogs
        volume_model = load_model(use_fixed_path=False)
        scaler = load_scaler(use_fixed_path=False)

    #read current belt's 'initial wear', 'removed_volume', 'RPM' and predict 'Force' and 'grind_time'
    base_data = {
        'grind_time': 12.0,
        'avg_rpm': 9000.0,
        'avg_force': 6,
        'initial_wear': 30000000
    }

    # Create a DataFrame
    grind_data = pd.DataFrame([base_data])
    base_forces = [2, 3, 4, 5, 7, 9, 12]
    base_rpms = [6000, 7000, 8000, 8500, 9000]
    sensitivity_analysis(volume_model, scaler, grind_data)

if __name__ == "__main__":
    main()