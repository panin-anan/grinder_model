import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

from data_manager import DataManager

def preprocess_data(data, target_column, n_bootstrap=5):
    #Preprocess the data by splitting into features and target and then scaling.

    X = data.drop(columns=target_column)
    y = data[target_column]

    bootstrap_samples = []

    for _ in range(n_bootstrap):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=np.random.randint(0,100))

        # Feature scaling
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

        bootstrap_samples.append((X_train, X_test, y_train, y_test, scaler))

    return bootstrap_samples

def train_multi_svr_with_grid_search(X_train, y_train):
    """
    Train a Support Vector Regressor (SVR) for multi-output regression.
    Wrap the SVR with MultiOutputRegressor to handle multiple targets.
    """
    # Define the parameter grid
    param_grid = {
        'estimator__C': [100, 200, 500, 1000, 2000],
        'estimator__gamma': [0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
        'estimator__epsilon': [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5],
        'estimator__kernel': ['rbf']
    }


    # Initialize SVR model
    svr = SVR()
    multioutput_svr = MultiOutputRegressor(svr)

    # Use GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(multioutput_svr, param_grid, cv=5, scoring='neg_mean_absolute_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score (MAE):", -grid_search.best_score_)

    # Get the best model
    best_model = grid_search.best_estimator_


    return best_model

def train_gbm_with_grid_search(X_train, y_train):
    """
    Train a Gradient Boosting Regressor for multi-output regression.
    Use GridSearchCV to search for the best hyperparameters.
    """
    # Define the parameter grid for GradientBoostingRegressor
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of boosting stages
        'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Learning rate
        'max_depth': [3, 4, 5, 6]  # Depth of the individual trees
    }

    # Initialize the Gradient Boosting Regressor
    gbr = GradientBoostingRegressor()


    # Use GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(gbr, param_grid, cv=5, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best parameters:", grid_search.best_params_)

    # Get the best model
    best_model = grid_search.best_estimator_

    return best_model

def train_multi_rf_with_grid_search(X_train, y_train):
    """
    Train a Random Forest Model
    """
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200, 300],  # Number of trees in the forest
        'max_depth': [5, 10, 20],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [2, 4, 8],    # Minimum number of samples required to be at a leaf node
        'bootstrap': [True, False]        # Whether bootstrap samples are used when building trees
    }

    # Initialize RandomForestRegressor
    rf = RandomForestRegressor(random_state=42)
    y_train = y_train.values.ravel()
    # Use GridSearchCV to search for the best hyperparameters
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best parameters:", grid_search.best_params_)

    # Get the best model
    best_model = grid_search.best_estimator_


    return best_model

def train_polynomial_regression_with_grid_search(X_train, y_train):
    """
    Train a Polynomial Regression model for multi-output regression.
    Use PolynomialFeatures with LinearRegression and GridSearchCV to search for the best degree.
    """
    # Define the parameter grid for the polynomial degree
    param_grid = {
        'poly__degree': [1, 2, 3, 4, 5, 6, 8, 10]  # Adjust the range based on your needs
    }

    # Create a pipeline that adds polynomial features and applies linear regression
    pipeline = Pipeline([
        ('poly', PolynomialFeatures()),  # Create polynomial features
        ('linear', LinearRegression())   # Fit linear regression
    ])

    # Use GridSearchCV to search for the best polynomial degree
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_root_mean_squared_error', verbose=1, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Print the best parameters found by GridSearchCV
    print("Best parameters:", grid_search.best_params_)

    # Get the best model
    best_model = grid_search.best_estimator_

    return best_model


def evaluate_model(model, X_test, y_test, OG_grind_data):
    y_pred = model.predict(X_test)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    #take indices with specific area in OG_grind_data
    special_grind_areas = OG_grind_data['grind_area'].unique()
    highlight_masks = {
        grind_area: y_test['index'].isin(OG_grind_data[OG_grind_data['grind_area'] == grind_area].index)
        for grind_area in special_grind_areas
    }

    y_test_no_index = y_test.drop(columns=['index'])

    # Evaluate the model with Mean Squared Error and R^2 Score
    mse = mean_squared_error(y_test.drop(columns=['index']), y_pred)
    rmse = np.sqrt(mse) 
    mean_abs = mean_absolute_error(y_test.drop(columns=['index']), y_pred)
    r2 = r2_score(y_test.drop(columns=['index']), y_pred)

    print(f"Mean Absolute Error: {mean_abs}")
    print(f"RMS Error: {rmse}")
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Plot actual vs predicted for each output
    plt.figure(figsize=(12, 6))
    
    for i, col in enumerate(y_test_no_index.columns):
        plt.subplot(1, len(y_test_no_index.columns), i + 1)

        # Plot each specific grind_area with a different color and include count in the label
        colors = plt.cm.tab10(np.linspace(0, 1, len(special_grind_areas)))   # Define colors for each special grind_area
        for j, (grind_area, mask) in enumerate(highlight_masks.items()):
            count = mask.sum()  # Count the number of points for this grind_area
            plt.scatter(y_test_no_index[col][mask], y_pred[mask, i], color=colors[j], 
                        label=f'grind_area = {grind_area} (n={count})', alpha=0.7)

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

def train_and_select_best_model(grind_data, target_columns):
    # Preprocess the data (generate bootstrap samples)
    bootstrap_samples = preprocess_data(grind_data, target_columns)

    models = []
    maes = []
    scalers = []

    # Train models on bootstrap samples and calculate MAE for each
    for i, (X_train, X_test, y_train, y_test, scaler) in enumerate(bootstrap_samples):
        # Train a model on each bootstrap sample
        model = train_multi_rf_with_grid_search(X_train, y_train.drop(columns=['index']))

        # Evaluate model and calculate MAE
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test.drop(columns=['index']), y_pred)
        
        # Store the model and its MAE
        models.append(model)
        maes.append(mae)
        scalers.append(scaler)

        print(f"Model {i+1} - MAE: {mae}")

    # Calculate the average MAE across all models
    avg_mae = np.mean(maes)
    print(f"Average MAE across all models: {avg_mae}")

    # Find the model with the MAE closest to the average MAE
    closest_index = np.argmin([abs(mae - avg_mae) for mae in maes])
    best_model = models[closest_index]
    best_model_mae = maes[closest_index]
    best_scaler = scalers[closest_index]

    print(f"Selected model {closest_index+1} with MAE closest to average: {best_model_mae}")

    # Retrieve the corresponding test set for the best model
    _, best_X_test, _, best_y_test, _ = bootstrap_samples[closest_index]

    return best_model, best_scaler, best_X_test, best_y_test

def save_model(model, scaler, folder_name='saved_models', modelname='svr_model.pkl', scalername='scaler.pkl'):
    # Get the current working directory
    current_dir = os.getcwd()

    # Create the full path by joining the current directory with the folder name
    folder_path = os.path.join(current_dir, 'src/grinder_model', folder_name)

    # Create the folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    # Create the full filepath to save the model
    model_filepath = os.path.join(folder_path, modelname)
    scaler_filepath = os.path.join(folder_path, scalername)

    if os.path.exists(model_filepath) or os.path.exists(scaler_filepath):
        overwrite = input(f"Files '{model_filepath}' or '{scaler_filepath}' already exist. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Model and scaler save operation cancelled.")
            return

    # Save the model to the specified filepath
    joblib.dump(model, model_filepath)
    joblib.dump(scaler, scaler_filepath)
    print(f"Model saved to {model_filepath}")
    print(f"Scaler saved to {scaler_filepath}")

def load_model(folder_name='saved_models', filename='svr_model.pkl'):
    # Get the current working directory
    current_dir = os.getcwd()

    # Create the full path by joining the current directory with the folder name
    folder_path = os.path.join(current_dir, 'src/grinder_model', folder_name)

    # Create the full filepath to load the model from
    filepath = os.path.join(folder_path, filename)

    # Load the model
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model


def main():
    #read grind data
    data_manager = DataManager()
    grind_data = data_manager.load_data()

    #filter out points that has high mad_rpm, material removal of less than 5, duplicates, failure msg detected
    grind_data = data_manager.filter_grind_data()
    grind_data['index'] = grind_data.index
    OG_grind_data = grind_data

    print(grind_data)

    #drop unrelated columns
    related_columns = ['grind_time', 'avg_rpm', 'avg_force', 'grind_area', 'initial_wear', 'removed_material', 'index']
    grind_data = grind_data[related_columns]

    #desired output
    target_columns = ['removed_material', 'index']

    # Train and select best model out of specified number of bootstrap
    best_model, best_scaler, best_X_test, best_y_test = train_and_select_best_model(grind_data, target_columns)

    # Optionally, evaluate the model on the test set
    #evaluate_model(best_model, X_train, y_train)
    evaluate_model(best_model, best_X_test, best_y_test, OG_grind_data)
 
    #save model
    save_model(best_model, best_scaler, folder_name='saved_models', modelname='volume_model_rf_W13_withgeom.pkl', scalername='volume_scaler_rf_W13_withgeom.pkl')



if __name__ == "__main__":
    main()