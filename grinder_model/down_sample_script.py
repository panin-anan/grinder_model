import pandas as pd

def downsample_data(input_file, output_file, num_points=40):
    # Load the original data from the CSV file
    data = pd.read_csv(input_file)
    
    # Check if there are at least 100 points in the data
    if len(data) < 100:
        raise ValueError("The dataset has fewer than 100 data points.")
    
    # Downsample from 100 to 40 points using random sampling
    downsampled_data = data.sample(n=num_points, random_state=42).sort_index()
    
    # Save the downsampled data to a new CSV file
    downsampled_data.to_csv(output_file, index=False)
    print(f"Downsampled data saved to {output_file}")

input_file = 'Test_data/Grind_data/data_gathering 5_plus_volume_fix_2_add_P&A_wear.csv'
output_file = 'Test_data/Grind_data/data_gathering 5_plus_volume_fix_2_add_P&A_wear_DownSampled.csv'
downsample_data(input_file, output_file)
