import pathlib
from scipy.optimize import minimize
import pandas as pd
import numpy as np

from volume_predictor_svr import load_model, load_scaler


def volume_mismatch_penalty(x, volume, wear, model, scaler, rpm):
    predicted_volume = predict_volume(x[0], rpm, x[1], wear, model, scaler)
    return (volume - predicted_volume) ** 2


def predict_volume(force, rpm, time, wear, model, scaler):
    input_data_dict = {
                       'grind_time': [time],
                       'avg_rpm': [rpm],
                       'avg_force': [force],
                       'initial_wear': [wear]
    }
    input_df = pd.DataFrame(input_data_dict)
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
    predicted_volume = model.predict(input_scaled)[0][0]
    return predicted_volume


def generate_settings(volume, wear, model, scaler, rpm=11000):
    
    # x = [force, time]
    x = [5, 10]
    min_f, max_f = 3, 9
    min_t, max_t = 5, 20
    result = minimize(volume_mismatch_penalty, x, args=(volume, wear, model, scaler, rpm),
                      bounds=((min_f, max_f), (min_t, max_t)))

    settings = {
                'force': result.x[0],
                'time': result.x[1],
                'rpm': rpm,
    }
    
    predicted_volume = predict_volume(settings['force'], settings['rpm'], settings['time'], wear, model, scaler)
    return settings, predicted_volume



if __name__ == '__main__':

    # TODO need an extra layer of setpoint to acutal rpm 
    # TODO model seems to underpredict material removal

    model_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_model_svr_V1.pkl'
    scaler_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_scaler_svr_V1.pkl'

    grind_model = load_model(use_fixed_path=True, fixed_path=model_path)
    scaler = load_scaler(use_fixed_path=True, fixed_path=scaler_path)

    removed_material = np.arange(10, 100, 5)
    wear_range = np.linspace(1e6, 1e7, 3)

    for vol in removed_material:
        for wear in wear_range:

            grind_settings, predicted_volume_loss = generate_settings(vol, wear, grind_model, scaler, 10000)

            print(f'\n\nSettings:\n  force: {grind_settings["force"]}\n  rpm:{grind_settings["rpm"]}\n  time: {grind_settings["time"]}')
            print(f'Removed material\n  input: {vol}\n  predicted: {predicted_volume_loss}')



