import pandas as pd
import tkinter as tk
from tkinter import filedialog

class DataManager:
    def __init__(self):
        self.grind_data = None

    def load_data(self):
        # Initial file selection
        file_path = self.open_file_dialog()
        if not file_path:
            print("No file selected. Exiting.")
            return None

        # Load the first file's data
        self.grind_data = pd.read_csv(file_path)

        # Loop to add more files if the user chooses
        while True:
            another_file_path = self.open_file_dialog()
            if not another_file_path:
                print("No more files selected. Moving forward with concatenated data.")
                break

            # Load and concatenate additional data
            additional_data = pd.read_csv(another_file_path)
            self.grind_data = pd.concat([self.grind_data, additional_data], ignore_index=True)
            print("File added successfully. Would you like to add another file?")

        print("All files loaded and concatenated.")
        return self.grind_data

    def filter_grind_data(self):
        if self.grind_data is None:
            print("No data loaded to filter.")
            return None

        # Delete rows where removed_material is less than 3
        self.grind_data = self.grind_data[self.grind_data['removed_material'] >= 3]

        # Filter out points with mad_rpm greater than 1000
        self.grind_data = self.grind_data[self.grind_data['mad_rpm'] <= 1000]

        # Filter out points with mad_rpm greater than 1000
        self.grind_data = self.grind_data[self.grind_data['avg_rpm'] < 11000.0]

        # Filter out points with mad_rpm greater than 1000
        #self.grind_data = self.grind_data[self.grind_data['feed_rate_setpoint'] > 11.0]

        # Filter out rows where avg_rpm is less than half of rpm_setpoint
        self.grind_data = self.grind_data[self.grind_data['avg_rpm'] >= self.grind_data['rpm_setpoint'] / 2]

        # Remove rows with any failure messages
        self.grind_data = self.grind_data[pd.isna(self.grind_data['failure_msg'])]

        # Check for duplicate 'removed_material' values
        duplicate_removed_material = self.grind_data[self.grind_data.duplicated(subset=['removed_material'], keep=False)]
        if not duplicate_removed_material.empty:
            print("Warning: Duplicate 'removed_material' values found:")
            print(duplicate_removed_material)
        self.grind_data = self.grind_data.drop_duplicates(subset=['removed_material'], keep='first')
        
        #print(self.grind_data)
        return self.grind_data

    def open_file_dialog(self):
        # Create a Tkinter window
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Open file dialog and return selected file path
        file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])
        return file_path