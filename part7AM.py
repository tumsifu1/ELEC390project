import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import joblib
from scipy.stats import skew

# Function to apply moving average filter and extract features
def data_feature_extraction(windows, w_size=5):
    filtered_data = np.zeros((windows.shape[0], windows.shape[1] - w_size + 1, windows.shape[2]))
    features_list = []

    for i in range(windows.shape[0]):
        # Apply moving average filter
        sma = pd.DataFrame(windows[i]).rolling(w_size).mean().dropna().values
        filtered_data[i] = sma
        
        features_window = []
        for j in range(windows.shape[2]):
            window_data = sma[:, j]
            max_val, min_val, mean_val, median_val = np.max(window_data), np.min(window_data), np.mean(window_data), np.median(window_data)
            range_val, var_val, skew_val, rms_val = max_val - min_val, np.var(window_data), skew(window_data), np.sqrt(np.mean(window_data ** 2))
            kurt_val, std_val = np.mean((window_data - mean_val) ** 4) / (var_val ** 2), np.std(window_data)
            features_window.extend([max_val, min_val, range_val, mean_val, median_val, var_val, skew_val, rms_val, kurt_val, std_val])
        features_list.append(features_window)

    return np.array(features_list)

def load_input_file():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            input_data = pd.read_csv(file_path)
            data = input_data.drop(columns=['Time (s)'])

            window_time = 500
            num_rows = len(data)
            num_windows = num_rows // window_time
            num_rows = num_windows * window_time
            data = data.iloc[:num_rows]

            data_windows = [data.iloc[i:i+window_time] for i in range(0, len(data), window_time)]
            data_array = np.stack(data_windows)
            data_features = data_feature_extraction(data_array)

            column_labels = [f'{axis}_{col}' for axis in 'xyztotal' for col in ['max_val', 'min_val', 'range_val', 'mean_val', 'median_val', 'var_val', 'skew_val', 'rms_val', 'kurt_val', 'std_val']]
            dataset = pd.DataFrame(data_features, columns=column_labels)
            
            missing_columns = 800 - dataset.shape[1]
            for i in range(missing_columns):
                dataset[f'extra_{i}'] = 0
            
            X_combined = dataset.values

            clfCombined = joblib.load('classifier.joblib')
            Y_predicted = clfCombined.predict(X_combined)
            Y_output = np.reshape(Y_predicted, (-1, 1))

            output_data = pd.DataFrame(np.hstack((Y_output, X_combined)), columns=['activity'] + column_labels + [f'extra_{i}' for i in range(missing_columns)])
            output_data['activity'] = np.where(output_data['activity'] == 0, 'walking', 'jumping')
            output_data.to_csv('output_data.csv')
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def create_widgets(root):
    main_frame = tk.Frame(root)
    main_frame.pack(fill=tk.BOTH, expand=True)

    load_input_button = tk.Button(main_frame, text="Load Input File", command=load_input_file)
    load_input_button.grid(row=0, column=0, padx=10, pady=10)

   # exit_button = tk.Button(main_frame, text="Exit", command=root.quit)
    #exit_button.grid(row=1, column=0, padx=10, pady= 10)

def main():
    root = tk.Tk()
    root.title("Activity Classifier")
    root.geometry("300x200")

    create_widgets(root)

    root.mainloop()

if __name__ == "__main__":
    main()

