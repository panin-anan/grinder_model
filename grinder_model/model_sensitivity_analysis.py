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
        pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_model_svr_V1_OG_nogeom.pkl',
        pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_model_svr_W13_withgeom.pkl'
    ]
    
    scaler_paths = [
        pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_scaler_svr_V1_OG_nogeom.pkl',
        pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_scaler_svr_W13_withgeom.pkl'
    ]

    # Load all models and scalers
    models = [load_model(use_fixed_path=True, fixed_path=path) for path in model_paths]
    scalers = [load_scaler(use_fixed_path=True, fixed_path=path) for path in scaler_paths]

    # Define base data for sensitivity analysis (excluding avg_force to be calculated)
    base_data_template = {
        'grind_time': 12.0,
        'avg_rpm': 9000.0,
        'avg_force': 6.0,
        'grind_area': 50,
        'initial_wear': 30000000
    }

    # Variables to vary
    variables = ['grind_time', 'avg_rpm', 'avg_force', 'grind_area', 'initial_wear']
    ranges = {
        'grind_time': np.linspace(5, 20, 100),
        'avg_rpm': np.linspace(8500, 10500, 100),
        'avg_force': np.linspace(3, 9, 100),
        'grind_area': np.linspace(50, 150, 100),
        'initial_wear': np.linspace(10000000, 45000000, 100)
    }

    # Create subplots
    fig, axes = plt.subplots(1, 5, figsize=(20, 6), sharey=True)

    for i, variable in enumerate(variables):
        values = ranges[variable]

        # Plot sensitivities for each model
        for j, (model, scaler) in enumerate(zip(models, scalers), 1):
            sensitivities = []
            for value in values:
                test_data = base_data_template.copy()
                test_data[variable] = value
                grind_data = pd.DataFrame([test_data])
                predicted_volume = predict_volume(model, scaler, grind_data.iloc[0])
                sensitivities.append(predicted_volume)

            # Plot results with unique styles for each model
            axes[i].plot(values, sensitivities, label=f'Model {j}', linestyle='-' if j == 1 else '--')

        
        # Customize each subplot
        axes[i].set_xlabel(variable)
        axes[i].set_title(f'{variable} Sensitivity')
        axes[i].grid(True)

        # Add legend only to the first plot
        if i == 0:
            axes[i].legend()

    # Customize the entire figure
    axes[0].set_ylabel('Predicted Grinded Volume')
    fig.suptitle('Sensitivity Analysis Across Variables', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    
if __name__ == "__main__":
    main()