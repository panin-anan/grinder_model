import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import joblib
import pathlib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from volume_model_svr import evaluate_model, evaluate_model_time_tag

from data_manager import DataManager

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

def evaluate_model_feed_rate_tag(model, X_test, y_test, OG_grind_data):
    y_pred = model.predict(X_test)
    
    # Round feed_rate values to group similar areas
    OG_grind_data['rounded_feed_rate'] = OG_grind_data['feed_rate_setpoint'].round()
    rounded_feed_rates = OG_grind_data['rounded_feed_rate'].unique()
    
    # Create masks for each rounded feed_rate
    highlight_masks = {
        rounded_feed_rate: y_test['index'].isin(OG_grind_data[OG_grind_data['rounded_feed_rate'] == rounded_feed_rate].index)
        for rounded_feed_rate in rounded_feed_rates
    }

    y_test_no_index = y_test.drop(columns=['index', 'feed_rate_setpoint'])

    # Evaluate the model with Mean Squared Error and R^2 Score
    mse = mean_squared_error(y_test_no_index, y_pred)
    rmse = np.sqrt(mse) 
    mean_abs = mean_absolute_error(y_test_no_index, y_pred)
    r2 = r2_score(y_test_no_index, y_pred)

    print(f"Mean Absolute Error: {mean_abs}")
    print(f"RMS Error: {rmse}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Plot actual vs predicted for each output
    plt.figure(figsize=(12, 6))
    
    for i, col in enumerate(y_test_no_index.columns):
        plt.subplot(1, len(y_test_no_index.columns), i + 1)

        # Plot each rounded feed_rate with a different color and include count in the label
        colors = plt.cm.tab10(np.linspace(0, 1, len(rounded_feed_rates)))  # Define colors for each rounded_feed_rate
        for j, (rounded_feed_rate, mask) in enumerate(highlight_masks.items()):
            count = mask.sum()  # Count the number of points for this feed_rate
            plt.scatter(y_test_no_index[col][mask], y_pred[mask, i], color=colors[j], 
                        label=f'feed_rate ≈ {rounded_feed_rate} (n={count})', alpha=0.7)

        # Plot all other points
        all_highlighted_mask = np.logical_or.reduce(list(highlight_masks.values()))
        other_count = (~all_highlighted_mask).sum()
        plt.scatter(y_test_no_index[col][~all_highlighted_mask], y_pred[~all_highlighted_mask, i], 
                    color='blue', label=f'Other points (n={other_count})', alpha=0.5)

        # Set axis limits to be the same
        min_val = min(min(y_test_no_index[col]), min(y_pred[:, i]))
        max_val = max(max(y_test_no_index[col]), max(y_pred[:, i]))
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)

        # Plot reference diagonal line for perfect prediction
        plt.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--')

        plt.xlabel(f"Actual {col}")
        plt.ylabel(f"Predicted {col}")
        plt.title(f"Actual vs Predicted {col}")
        plt.legend()

    plt.tight_layout()
    plt.show()

def get_user_choice():
    print("Choose an option:")
    print("1. Use manual data input")
    print("2. Load CSV data for evaluation")
    choice = input("Enter 1 or 2: ")
    return choice


def main():

    user_choice =  get_user_choice()

    #get grind model
    use_fixed_model_path = True# Set this to True or False based on your need
    
    if use_fixed_model_path:
        # Specify the fixed model and scaler paths
        fixed_model_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_model_svr_W13_withgeom.pkl'
        fixed_scaler_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_scaler_svr_W13_withgeom.pkl'
        
        grind_model = load_model(use_fixed_path=True, fixed_path=fixed_model_path)
        scaler = load_scaler(use_fixed_path=True, fixed_path=fixed_scaler_path)
    else:
        # Load model and scaler using file dialogs
        grind_model = load_model(use_fixed_path=False)
        scaler = load_scaler(use_fixed_path=False)

    if user_choice == "1":
        #read current belt's 'initial wear', 'removed_volume', 'RPM' and predict 'Force' and 'grind_time'
        rpm_range = np.arange(9000, 9510, 500)  # from 8500 to 10000 in steps of 500
        force_range = np.arange(5, 5.1, 1)  # from 3 to 9 in steps of 1
        time_range = np.arange(12.5, 12.6, 0.5)
        grind_area = 50
        initial_wear = 30000000
        feed_rate = 10
        num_pass = 8
        pass_length = 100
        belt_width = 25

        for avg_rpm in rpm_range:
           for avg_force in force_range:
               for grind_time in time_range:
                    # Create a DataFrame to store the input data
                    input_data_dict = {
                        'grind_time': [grind_time],
                        'avg_rpm': [avg_rpm],
                        'avg_force': [avg_force],
                        'grind_area': [grind_area],
                        'initial_wear': [initial_wear]
                    }
                    input_df = pd.DataFrame(input_data_dict)
                    input_scaled = scaler.transform(input_df)
                    input_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)
                    # Predict volume
                    predicted_volume = grind_model.predict(input_scaled)
                    predicted_total_vol = predicted_volume[0]*pass_length/belt_width
                    print(f"RPM: {avg_rpm}, Force: {avg_force}N, grind_area: {grind_area}mm^2, Grind Time: {grind_time} sec --> Predicted Removed Volume: {predicted_volume[0]}")
                    #print(f"Grind Time: {grind_time} sec --> Feed_rate: {feed_rate} mm/s, Num_pass: {num_pass}, predicted_total_volume: {predicted_total_vol}")
    if user_choice == "2":    
        #load test data and evaluate model
        #read grind data
        data_manager = DataManager()
        grind_data = data_manager.load_data()
    
        #filter out points that has high mad_rpm, material removal of less than 5, duplicates, failure msg detected
        grind_data = data_manager.filter_grind_data()
        grind_data['index'] = grind_data.index
        grind_data['grind_time'] = grind_data['contact_time'] / 4
        OG_grind_data = grind_data                      #for tagging different parameter value settings in colors
    
        print(grind_data)
    
        #drop unrelated columns
        related_columns = ['grind_time', 'avg_rpm', 'avg_force', 'grind_area', 'initial_wear', 'removed_material', 'index', 'feed_rate_setpoint']
        grind_data = grind_data[related_columns]
    
        #desired output
        target_columns = ['removed_material', 'index', 'feed_rate_setpoint']
    
        # Preprocess the data (train the model using the CSV data, for example)
        X_test_scaled, y_test = preprocess_test_data(grind_data, target_columns, scaler)
    
        evaluate_model_feed_rate_tag(grind_model, X_test_scaled, y_test, OG_grind_data)
    
    
if __name__ == "__main__":
    main()