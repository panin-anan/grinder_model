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

def sensitivity_analysis_wear_forces(model, scaler, base_data, base_forces):
    """
    Perform sensitivity analysis on the trained model by varying initial wear across different base forces.
    Parameters:
        model: Trained SVR model.
        scaler: Scaler used to preprocess the input data.
        base_data: A single data point representing the manually input base values.
        base_forces: List of base forces to use for sensitivity analysis.
    """
    # Select the base data point for sensitivity analysis
    base_data = base_data.iloc[0]  # Use the first (and only) row for sensitivity analysis

    # Variables to test
    wears = np.linspace(base_data['initial_wear'] * 0, base_data['initial_wear'] * 1.5, 100)  # Vary initial wear up to 300%

    # Function to make predictions after scaling inputs
    def predict_volume(data_row):
        # Ensure the data_row is in the form of a DataFrame with proper column names
        if isinstance(data_row, pd.Series):
            data_row = pd.DataFrame([data_row])

        # Scale the input data using the scaler
        scaled_row = scaler.transform(data_row)
        
        # Predict the volume using the trained model
        return model.predict(scaled_row)[0]

    # Plot sensitivity analysis results
    plt.figure(figsize=(10, 6))

    # Loop through each base force and calculate wear sensitivities
    for base_force in base_forces:
        # Store results for wear sensitivity
        wear_sensitivities = []
        for wear in wears:
            test_data = base_data.copy()
            test_data['avg_force'] = base_force  # Set the base force for this iteration
            test_data['initial_wear'] = wear  # Vary initial wear
            predicted_volume = predict_volume(test_data)
            wear_sensitivities.append(predicted_volume)

        # Plot the wear sensitivities for this base force
        plt.plot(wears, wear_sensitivities, label=f'Force: {base_force}')

    # Customize the plot
    plt.xlabel('Initial Wear')
    plt.ylabel('Predicted Grinded Volume')
    plt.title('Sensitivity to Initial Wear for Different Base Forces')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def predict_volume(model, scaler, data_row):
    """Helper function to scale data and predict volume with the model."""
    # Ensure the data_row is in the form of a DataFrame with proper column names
    if isinstance(data_row, pd.Series):
        data_row = pd.DataFrame([data_row])

    # Reorder columns to match the scaler's expected input during fit
    data_row = data_row[scaler.feature_names_in_]

    # Scale and predict
    scaled_row = scaler.transform(data_row)
    return model.predict(scaled_row)[0]


def main():
    # Define paths for the three models and scalers
    model_paths = [
        pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_model_svr_V1_OG_withgeom.pkl',
        pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_model_svr_W13_withgeom.pkl'
    ]
    
    scaler_paths = [
        pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_scaler_svr_V1.pkl',
        pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_scaler_svr_W13_withgeom.pkl'
    ]

    # Load all models and scalers
    models = [load_model(use_fixed_path=True, fixed_path=path) for path in model_paths]
    scalers = [load_scaler(use_fixed_path=True, fixed_path=path) for path in scaler_paths]

    grind_areas = [50, 50]
    target_pressure = 0.09  # Adjust this value based on your application
    # Define base data for sensitivity analysis (excluding avg_force to be calculated)
    base_data_template = {
        'grind_time': 12.0,
        'avg_rpm': 9000.0,
        'initial_wear': 30000000
    }

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Perform sensitivity analysis for each model and plot
    for i, (model, scaler, grind_area) in enumerate(zip(models, scalers, grind_areas), 1):
        # Calculate force to achieve the target pressure for this model's grind area
        base_force = target_pressure * grind_area  # Force = Pressure * Area

        # Set up base data with the specific grind area and calculated force for this model
        base_data = base_data_template.copy()
        base_data['grind_area'] = grind_area
        base_data['avg_force'] = base_force
        grind_data = pd.DataFrame([base_data])

        # Run sensitivity analysis on wear for each model with the calculated force
        wears = np.linspace(base_data['initial_wear'] * 0, base_data['initial_wear'] * 1.5, 100)
        
        wear_sensitivities = []
        for wear in wears:
            test_data = grind_data.iloc[0].copy()
            test_data['initial_wear'] = wear
            predicted_volume = predict_volume(model, scaler, test_data)
            wear_sensitivities.append(predicted_volume)

        # Plot with model label for each grind area and corresponding force
        plt.plot(wears, wear_sensitivities, label=f'Model {i} (Grind Area: {grind_area}, Force: {base_force:.2f}N)')

    # Customize the plot for comparison
    plt.xlabel('Initial Wear')
    plt.ylabel('Predicted Grinded Volume')
    plt.title('Sensitivity to Initial Wear at Same Pressure Across Models')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()