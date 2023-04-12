import numpy as np
import pandas as pd
from scipy.stats import skew
import joblib
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
import os

# global variables
filePath = ""
fileName = ""
output_data = None
Y_clf_prob_combined = None


# feature extraction function called in the process data function
def data_feature_extraction(windows):
    w_size = 5

    filtered_data = np.zeros((windows.shape[0], windows.shape[1] - w_size + 1, windows.shape[2]))

    for i in range(windows.shape[0]):
        x_df = pd.DataFrame(windows[i, :, 0])
        y_df = pd.DataFrame(windows[i, :, 1])
        z_df = pd.DataFrame(windows[i, :, 2])
        total = windows[i, :, 3]

        x_sma = x_df.rolling(w_size).mean().values.ravel()
        y_sma = y_df.rolling(w_size).mean().values.ravel()
        z_sma = z_df.rolling(w_size).mean().values.ravel()

        x_sma = x_sma[w_size - 1:]
        y_sma = y_sma[w_size - 1:]
        z_sma = z_sma[w_size - 1:]
        total_sma = total[w_size - 1:]

        filtered_data[i, :, 0] = x_sma
        filtered_data[i, :, 1] = y_sma
        filtered_data[i, :, 2] = z_sma
        filtered_data[i, :, 3] = total_sma

    windows = filtered_data
    features = np.zeros((windows.shape[0], 10, 4))

    for i in range(windows.shape[2]):
        for j in range(windows.shape[0]):
            window_data = windows[j, :, i]

            max_val = np.max(window_data)
            min_val = np.min(window_data)
            range_val = max_val - min_val
            mean_val = np.mean(window_data)
            median_val = np.median(window_data)
            var_val = np.var(window_data)
            skew_val = skew(window_data)
            rms_val = np.sqrt(np.mean(window_data ** 2))
            kurt_val = np.mean((window_data - np.mean(window_data)) ** 4) / (np.var(window_data) ** 2)
            std_val = np.std(window_data)

            features[j, :, i] = (max_val, min_val, range_val, mean_val, median_val, var_val, skew_val, rms_val,
                                 kurt_val, std_val)

    x_feature = features[:, :, 0]
    y_feature = features[:, :, 1]
    z_feature = features[:, :, 2]
    total_feature = features[:, :, 3]

    all_features = np.concatenate((x_feature, y_feature, z_feature, total_feature), axis=0)

    labels = np.concatenate((np.ones((x_feature.shape[0], 1)),
                             2 * np.ones((y_feature.shape[0], 1)),
                             3 * np.ones((z_feature.shape[0], 1)),
                             4 * np.ones((total_feature.shape[0], 1))), axis=0)

    all_features = np.hstack((all_features, labels))

    return all_features


# gets the name of the file in the input field
def getname():
    global fileName
    fileName = inputtxt.get("1.0", "end-1c")


# clears the input and output field
def clear_selection():
    inputtxt.delete("1.0", "end")
    selected_file_label.config(text="")


# allows you to plot scatter of the outputs
def view_plots(output_data):
    global Y_clf_prob_combined

    # Define x-axis data (window number)
    x_data = output_data.index

    # get probabilities

    # Define y-axis data (walking and jumping probabilities)
    y_data = Y_clf_prob_combined.reshape((-1, 2))

    # Define bar labels
    bar_labels = ['Walking Probability', 'Jumping Probability']

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(len(bar_labels)):
        ax.bar(x_data + i * 0.4, y_data[:, i], width=0.4, label=bar_labels[i])

    # Add activity labels above bars
    for i in range(len(x_data)):
        if output_data.iloc[i]['activity'] == 'walking':
            label = 'W'
            color = 'blue'
        else:
            label = 'J'
            color = 'orange'
        ax.text(x_data[i] - 0.1, max(y_data[i]) + 0.05, label, color=color, fontweight='bold', fontsize=12)

    # Set plot labels and legend
    ax.set_xlabel('Window Number')
    ax.set_ylabel('Probability')
    ax.set_xticks(x_data)
    ax.set_xticklabels(x_data + 1)
    ax.set_title('Walking and Jumping Probabilities by Window Number')
    ax.legend()
    ax.set_ylim([0, 1.2])

    # Show plot
    plt.show()

    if output_data is None:
        tk.messagebox.showwarning(title="Warning", message="Output data is empty. Please try again!")
    output_data = output_data.drop(index=0)


    walking_data = output_data[output_data['activity'] == 'walking']
    jumping_data = output_data[output_data['activity'] == 'jumping']

    plt.scatter(jumping_data['total_var_val'], jumping_data['total_mean_val'], c='blue', label='Jumping')
    plt.scatter(walking_data['total_var_val'], walking_data['total_mean_val'], c='red', label='walking')
    plt.xlabel('VAR')
    plt.ylabel('MEAN')
    plt.title('Scatter plot of mean vs range for the absolute acceleration of classified windows')
    plt.legend()
    plt.show()


# gets the path for the outputs
def getpath():
    global filePath
    global output_data

    filePath = filedialog.askopenfilename()
    selected_file_label.config(text=os.path.basename(filePath))


# process input the data and classifes it
def processData():
    global output_data
    global Y_clf_prob_combined

    if not filePath:
        tk.messagebox.showwarning(title="Warning", message="Please select a file before proceeding!")
        return
    if not fileName:
        tk.messagebox.showwarning(title="Warning", message="Please enter a name for the output file before proceeding!")
        return

    input_data = pd.read_csv(filePath)
    data = input_data.drop(columns=['Time (s)'])

    window_time = 500
    num_rows = len(data)
    num_windows = num_rows // window_time
    num_rows = num_windows * window_time
    data = data.iloc[:num_rows]

    data_windows = []
    for i in range(0, len(data), window_time):
        if i + window_time <= len(data):
            data_windows.append(data.iloc[i:i + window_time, :])

    data_array = np.stack(data_windows)

    data_features = data_feature_extraction(data_array)

    column_labels = np.array(
        ['max_val', 'min_val', 'range_val', 'mean_val', 'median_val', 'var_val', 'skew_val', 'rms_val', 'kurt_val',
         'std_val', 'measurement'])

    dataset = pd.DataFrame(data_features, columns=column_labels)

    featureNumber = 10

    xData = dataset[dataset.iloc[:, featureNumber] == 1]
    yData = dataset[dataset.iloc[:, featureNumber] == 2]
    zData = dataset[dataset.iloc[:, featureNumber] == 3]
    allData = dataset[dataset.iloc[:, featureNumber] == 4]

    X_xdata = xData.iloc[:, 0:-1]
    X_ydata = yData.iloc[:, 0:-1]
    X_zdata = zData.iloc[:, 0:-1]
    X_alldata = allData.iloc[:, 0:-1]

    X_combined = np.zeros((X_xdata.shape[0], 4 * featureNumber))
    for k in range(featureNumber):
        X_combined[:, k] = X_xdata.iloc[:, k]
        X_combined[:, k + featureNumber] = X_ydata.iloc[:, k]
        X_combined[:, k + (2 * featureNumber)] = X_zdata.iloc[:, k]
        X_combined[:, k + (3 * featureNumber)] = X_alldata.iloc[:, k]

    clfCombined = joblib.load('classifier.joblib')

    Y_predicted = clfCombined.predict(X_combined)
    Y_output = np.reshape(Y_predicted, (-1, 1))
    Y_clf_prob_combined = clfCombined.predict_proba(X_combined)

    column_labels = np.array(
        ['activity', 'x_max_val', 'x_min_val', 'x_range_val', 'x_mean_val', 'x_median_val', 'x_var_val', 'x_skew_val',
         'x_rms_val', 'x_kurt_val', 'x_std_val', 'y_max_val', 'y_min_val', 'y_range_val', 'y_mean_val', 'y_median_val',
         'y_var_val', 'y_skew_val', 'y_rms_val', 'y_kurt_val', 'y_std_val', 'z_max_val', 'z_min_val', 'z_range_val',
         'z_mean_val', 'z_median_val', 'z_var_val', 'z_skew_val', 'z_rms_val', 'z_kurt_val', 'z_std_val',
         'total_max_val', 'total_min_val', 'total_range_val', 'total_mean_val', 'total_median_val', 'total_var_val',
         'total_skew_val', 'total_rms_val', 'x_kurt_val', 'x_std_val'])

    output_data = pd.DataFrame(np.hstack((Y_output, X_combined)), columns=column_labels)

    output_data['activity'] = np.where(output_data['activity'] == 0, 'walking', 'jumping')

    output_data.to_csv(fileName)
    tk.messagebox.showinfo(title="Done", message="Output file has been saved!")
    view_plots_btn.config(state=tk.NORMAL)  # Enable the view_plots_btn button
    return output_data


# UI

window = tk.Tk()

# Set window background color to black
window.configure(bg='black')

window.title('Acceleration Classifier')

greeting = tk.Label(text="Welcome to the acceleration classifier!", fg='#7FDBFF', bg='black',
                    font=('Arial', 14, 'bold'), width=300, height=10)
greeting.pack()

btn = tk.Button(window, text="Select File", command=getpath, bg='white', fg='black', font=('Arial', 10, 'bold'),
                width=20)
btn.place(x=430, y=150)

inputGreeting = tk.Label(text="Name your output file!", fg='#7FDBFF', bg='black', font=('Arial', 12))
inputGreeting.pack()
inputGreeting.place(x=425, y=200)

inputtxt = tk.Text(window, height=1, width=21, bg='white', fg='black')
inputtxt.pack()
inputtxt.place(x=430, y=240)
submit = tk.Button(window, text="Submit", command=lambda: [getname(), processData()], bg='white', fg='black',
                   font=('Arial', 10, 'bold'), width=20)

clear_selection_btn = tk.Button(window, text="Clear Selection", command=clear_selection, bg='white', fg='black',
                                font=('Arial', 10, 'bold'), width=20)
clear_selection_btn.place(x=430, y=350)

progress_bar = ttk.Progressbar(window, orient="horizontal", length=300, mode="determinate")
progress_bar.place(x=350, y=450)

view_plots_btn = tk.Button(window, text="View Plots", command=lambda: view_plots(output_data), bg='white', fg='black',
                           font=('Arial', 10, 'bold'), width=20, state=tk.DISABLED)
view_plots_btn.place(x=430, y=400)

submit.place(x=430, y=300)
window.geometry("1000x600")

selected_file_label = tk.Label(window, text="", fg='#7FDBFF', bg='black')
selected_file_label.place(x=430, y=180)

window.mainloop()

window.title('Acceleration Classifier')

greeting = tk.Label(text="Welcome to the acceleration classifier!", fg='#7FDBFF', bg='black', font=('Arial', 14, 'bold'), width=300, height=10)
greeting.pack()

btn = tk.Button(window, text="Select File", command=getpath, bg='white', fg='black', font=('Arial', 10, 'bold'), width=20)
btn.place(x=430, y=150)

inputGreeting = tk.Label(text="Name your output file!", fg='#7FDBFF', bg='black', font=('Arial', 12))
inputGreeting.pack()
inputGreeting.place(x=425, y=200)

inputtxt = tk.Text(window, height=1, width=21, bg='white', fg='black')
inputtxt.pack()
inputtxt.place(x=430, y=240)
submit = tk.Button(window, text="Submit", command=lambda: [getname(), processData()], bg='white', fg='black', font=('Arial', 10, 'bold'), width=20)

clear_selection_btn = tk.Button(window, text="Clear Selection", command=clear_selection, bg='white', fg='black', font=('Arial', 10, 'bold'), width=20)
clear_selection_btn.place(x=430, y=350)

progress_bar = ttk.Progressbar(window, orient="horizontal", length=300, mode="determinate")
progress_bar.place(x=350, y=450)

view_plots_btn = tk.Button(window, text="View Plots", command=lambda: view_plots(output_data), bg='white', fg='black', font=('Arial', 10, 'bold'), width=20, state=tk.DISABLED)
view_plots_btn.place(x=430, y=400)

submit.place(x=430, y=300)
window.geometry("1000x600")

selected_file_label = tk.Label(window, text="", fg='#7FDBFF', bg='black')
selected_file_label.place(x=430, y=180)

window.mainloop()
