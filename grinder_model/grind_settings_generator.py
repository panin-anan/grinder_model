import pathlib
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import math

from volume_predictor_svr import load_model, load_scaler


def volume_mismatch_penalty(x, volume, wear, model, scaler, rpm, area):
    predicted_volume = predict_volume(x[0], area, rpm, x[1], wear, model, scaler)
    return (volume - predicted_volume) ** 2


def predict_volume(force, area, rpm, time, wear, model, scaler):
    input_data_dict = {
                       'grind_time': [time],
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
    
    # x = [force, time] initial guess value
    x = [5, 10]
    min_f, max_f = 3, 6
    min_t, max_t = 10, 20
    result = minimize(volume_mismatch_penalty, x, args=(volume, wear, model, scaler, rpm, area),
                      bounds=((min_f, max_f), (min_t, max_t)))

    # For cases where rpm setpoint and actual avg_rpm are not the same.
    # input_rpm_correction_data_dict = {
    #     'avg_force': [result.x[0]],
    #     'grind_area': [area],
    #     'rpm_setpoint': [rpm],
    #     'initial_wear': [wear]
    # }
    # input_df = pd.DataFrame(input_rpm_correction_data_dict)
    # input_scaled = rpm_scaler.transform(input_df)
    # input_scaled = pd.DataFrame(input_scaled, columns=input_df.columns)
    #predicted_avg_rpm = rpm_model.predict(input_scaled)

    settings = {
                'force': result.x[0],
                'time': result.x[1],
                'rpm': rpm,
    }
    predicted_volume = predict_volume(settings['force'], area, settings['rpm'], settings['time'], wear, model, scaler)
    
    if vol > 130:
        mrr = predicted_volume / settings['time']
        settings['time'] = volume / mrr
        predicted_volume = volume

    return settings, predicted_volume



if __name__ == '__main__':
    # RPM correction model is for when test setup is limited in flow rate, 
    # set rpm and actual avg__rpm will have noticable difference and rpm correction model need to be used.
    rpm_correction_model_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'rpm_correction_model_svr_W13.pkl'
    rpm_correction_scaler_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'rpm_correction_scaler_svr_W13.pkl'
    rpm_correction_model = load_model(use_fixed_path=True, fixed_path=rpm_correction_model_path)
    rpm_correction_scaler = load_scaler(use_fixed_path=True, fixed_path=rpm_correction_scaler_path)

    # load predictive model
    model_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_model_svr_W13_withgeom.pkl'
    scaler_path = pathlib.Path.cwd() / 'src' / 'grinder_model' / 'saved_models' / 'volume_scaler_svr_W13_withgeom.pkl'

    grind_model = load_model(use_fixed_path=True, fixed_path=model_path)
    grind_scaler = load_scaler(use_fixed_path=True, fixed_path=scaler_path)

    removed_material = [30]
    wear_range = [3e7]                          #np.linspace(1e6, 3e6, 2)
    belt_width = 25                          #in mm
    plate_thickness = 2                     #in mm
    belt_angle = 0                              #in degree
    set_rpm = 9500
    init_feed_rate = 20


    contact_width = belt_width * math.cos(math.radians(belt_angle))
    grind_area = belt_width * plate_thickness

    for vol in removed_material:
        for wear in wear_range:

            grind_settings, predicted_volume_loss = generate_settings(vol, wear, grind_model, grind_scaler, rpm_correction_model, rpm_correction_scaler, set_rpm, grind_area)
            init_num_pass = init_feed_rate * grind_settings["time"] / (belt_width)
            num_pass = np.round(init_num_pass)
            feed_rate = num_pass*belt_width  / grind_settings["time"]

            print(f'\n\nSettings set_rpm: {set_rpm}:\n  force: {grind_settings["force"]}\n  corrected_rpm:{grind_settings["rpm"]}\n  time: {grind_settings["time"]}\n feed_rate: {feed_rate} mm/s\n num_pass: {num_pass}')
            # print(f'Removed material Total for plate {total_path_length*1000} mm\n input: {vol_total}\n  predicted: {predicted_volume_loss*total_path_length/belt_width}')
            print(f'Removed material factored\n  input: {vol}\n  predicted: {predicted_volume_loss}')
            print(f"Removal depth [mm]: {predicted_volume_loss / (plate_thickness *  belt_width)}")





