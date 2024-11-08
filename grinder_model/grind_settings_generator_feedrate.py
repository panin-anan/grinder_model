import pathlib
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import math

from volume_predictor_svr import load_model, load_scaler


def volume_mismatch_penalty(x, volume, wear, model, scaler, rpm, area):
    predicted_volume = predict_volume(x[0], area, rpm, x[1], x[2], wear, model, scaler)
    return (volume - predicted_volume) ** 2

def volume_mismatch_penalty_fixed_num_pass(x, volume, wear, model, scaler, rpm, area, num_pass):
    predicted_volume = predict_volume(x[0], area, rpm, x[1], num_pass, wear, model, scaler)
    return (volume - predicted_volume) ** 2

def predict_volume(force, area, rpm, feed_rate, num_pass, wear, model, scaler):
    input_data_dict = {
                       'feed_rate': [feed_rate],
                       'num_pass': [num_pass],
                       'avg_rpm': [rpm],
                       'avg_force': [force],
                       'grind_area': [area],
                       'initial_wear': [wear]
    }
    input_df = pd.DataFrame(input_data_dict)
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
    predicted_volume = model.predict(input_scaled)[0][0]
    return predicted_volume


def generate_settings(volume, wear, model, scaler, rpm_model, rpm_scaler, rpm=11000, area=0.00005):
    
    # x = [force, feed_rate, num_pass]
    x = [5, 0.025, 3]
    min_f, max_f = 3, 9
    min_fr, max_fr = 0.010, 0.030
    min_pass, max_pass = 2, 4
    result = minimize(volume_mismatch_penalty, x, args=(volume, wear, model, scaler, rpm, area),
                      bounds=((min_f, max_f), (min_fr, max_fr), (min_pass, max_pass)))

    num_pass_rounded = round(result.x[2])
    # Run optimization again, but only for `force` and `feed_rate`
    x_fixed = [result.x[0], result.x[1]]
    result_fixed = minimize(volume_mismatch_penalty_fixed_num_pass, x_fixed, args=(volume, wear, model, scaler, rpm, area, num_pass_rounded),
                        bounds=((min_f, max_f), (min_fr, max_fr)))


    input_rpm_correction_data_dict = {
        'avg_force': [result_fixed.x[0]],
        'grind_area': [area],
        'rpm_setpoint': [rpm],
        'initial_wear': [wear]
    }
    input_df = pd.DataFrame(input_rpm_correction_data_dict)
    input_scaled = rpm_scaler.transform(input_df)
    input_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)
    predicted_avg_rpm = rpm_model.predict(input_scaled)


    settings = {
                'force': result_fixed.x[0],
                'feed_rate': result_fixed.x[1],
                'num_pass': num_pass_rounded,
                'rpm': predicted_avg_rpm,
    }
    predicted_volume = predict_volume(settings['force'], area, settings['rpm'], settings['feed_rate'], settings['num_pass'], wear, model, scaler)
    
    '''
    if vol > 130:
        mrr = predicted_volume / settings['time']
        settings['time'] = volume / mrr
        predicted_volume = volume
    '''

    return settings, predicted_volume



if __name__ == '__main__':

    rpm_correction_model_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'rpm_correction_model_svr_W13.pkl'
    rpm_correction_scaler_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'rpm_correction_scaler_svr_W13.pkl'
    rpm_correction_model = load_model(use_fixed_path=True, fixed_path=rpm_correction_model_path)
    rpm_correction_scaler = load_scaler(use_fixed_path=True, fixed_path=rpm_correction_scaler_path)

    model_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'FR_volume_model_svr_W13_withgeom.pkl'
    scaler_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'FR_volume_scaler_svr_W13_withgeom.pkl'

    grind_model = load_model(use_fixed_path=True, fixed_path=model_path)
    grind_scaler = load_scaler(use_fixed_path=True, fixed_path=scaler_path)

    removed_material = np.arange(40, 160, 20)
    wear_range = np.linspace(1e6, 3e6, 1)
    belt_width = 0.025                          #in m 
    plate_thickness = 0.002                     #in m
    belt_angle = 0                              #in degree
    total_path_length = 0.075                     #in m, only for total time estimation
    set_rpm = 10000

    #TODO implement contact width or make belt_width into contact_area
    contact_width = belt_width * math.cos(math.radians(belt_angle))
    grind_area = belt_width * plate_thickness

    for vol in removed_material:
        for wear in wear_range:
            grind_settings, predicted_volume_loss = generate_settings(vol, wear, grind_model, grind_scaler, rpm_correction_model, rpm_correction_scaler, set_rpm, grind_area)
            total_time = total_path_length * grind_settings["num_pass"] / grind_settings["feed_rate"]
            print(f'\n\nSettings:\n  force: {grind_settings["force"]}\n  avg_rpm:{grind_settings["rpm"]}\n Feed_rate: {grind_settings["feed_rate"] * 1000} mm/s\n num_pass:{grind_settings["num_pass"]}\n  total time: {total_time} s')
            print(f'Removed material\n  input: {vol}\n  predicted: {predicted_volume_loss}')






