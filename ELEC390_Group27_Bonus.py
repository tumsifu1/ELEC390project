# ELEC 390 Project Bonus

import numpy as np
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.common.by import By
import time
from scipy.stats import skew
import joblib
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# global var
driver = None
webdriver_path = None
ip_address = None

predicted_activity = None

clfCombined = joblib.load('classifier.joblib')


def feature_extract(window_data):
    # Compute the features
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

    # Store the features in the features array
    features = (max_val, min_val, range_val, mean_val, median_val, var_val, skew_val, rms_val, kurt_val, std_val)

    return features


def process_data_and_predict():
    global predicted_activity

    # Close the current UI
    window.destroy()

    # create new UI with predicted activity
    new_root = tk.Tk()
    new_root.title("Prediction")
    new_root.geometry("400x400")
    new_root.configure(bg="black")

    # Create the prediction label
    prediction_label = tk.Label(new_root, text="Waiting ...", bg="black", fg="white", font=("Helvetica", 35), wraplength=180, justify="center")
    prediction_label.pack(expand=True)

    # Update the main window
    new_root.update()

    window_size = 50  # Number of datapoints for each classification
    iterations = 15  # Number of times data gets classified
    time.sleep(10)  # Allows time for set up
    for n in range(iterations):
        x_data_array = np.zeros(window_size)
        y_data_array = np.zeros(window_size)
        z_data_array = np.zeros(window_size)
        total_data_array = np.zeros(window_size)

        # Initialize index to keep track of data points
        index = 0

        while True:
            try:
                element = WebDriverWait(driver, 50).until(
                    ec.presence_of_element_located((By.ID, "element6"))
                )

                # Collect data from elements 6, 7, 8
                soup = BeautifulSoup(driver.page_source, 'html.parser')

                value_x = float(
                    soup.find('div', {'id': 'element6'}).find('span', {'class': 'valueNumber'}).text.strip())
                value_y = float(
                    soup.find('div', {'id': 'element7'}).find('span', {'class': 'valueNumber'}).text.strip())
                value_z = float(
                    soup.find('div', {'id': 'element8'}).find('span', {'class': 'valueNumber'}).text.strip())
                value_total = np.sqrt(value_x ** 2 + value_y ** 2 + value_z ** 2)

                # Convert values to floats and store in data_array
                x_data_array[index] = value_x
                y_data_array[index] = value_y
                z_data_array[index] = value_z
                total_data_array[index] = value_total

                index += 1

                # Break loop
                if index == window_size:
                    break

            except KeyboardInterrupt:
                driver.quit()
                break

        x_data_features = feature_extract(x_data_array)
        y_data_features = feature_extract(y_data_array)
        z_data_features = feature_extract(z_data_array)
        total_data_features = feature_extract(total_data_array)

        X_combined = np.concatenate((x_data_features, y_data_features, z_data_features, total_data_features), axis=0)
        X_combined = X_combined.reshape(1, -1)

        Y_predicted = clfCombined.predict(X_combined)

        predicted_activity = Y_predicted[0]

        # update the UI window with the predicted activity
        if predicted_activity == 0:
            prediction_label.config(text="Walking")
            print('Walking')
        elif predicted_activity == 1:
            prediction_label.config(text="Jumping")
            print('Jumping')

        # update the UI window
        new_root.update()


# functions for UI
def get_webdriver_path():
    global webdriver_path
    webdriver_path = filedialog.askopenfilename()
    webdriver_path_label.config(text=webdriver_path)


def get_ip_address():
    global ip_address
    ip_address = ip_address_entry.get()
    if not ip_address:
        messagebox.showwarning("Warning", "Please enter a valid IP address.")

def submit():
    global driver, webdriver_path, ip_address, predicted_activity
    if webdriver_path and ip_address:
        driver = webdriver.Chrome(webdriver_path)
        driver.get(f'http://{ip_address}/')
        process_data_and_predict()

    else:
        messagebox.showwarning("Warning", "Please select a web driver and enter a valid IP address before submitting.")


def show_instructions():
    instructions = """1. Open the Phyphox app
2. Click on "Acceleration without G"
3. Click the button with three dots in the top right corner of the Phyphox interface and select 'Enable Remote access'
4. Select the location of your Chrome Web Driver into the UI text box 
5. Input the numbers from the URL provided by Phyphox into the UI text box and click the 'Enter' button
6. Click 'Submit'
7. Navigate to the 'Mutli' window
8. Wait a few seconds and then look on your screen to view the activity classification"""
    messagebox.showinfo("Instructions", instructions)

# UI
window = tk.Tk()
window.configure(bg='black')
window.title('Acceleration Classifier')

webdriver_path_label = tk.Label(window, width=40, text="", anchor='w', bg='white', relief='sunken')
webdriver_path_label.pack()
webdriver_path_label.place(x=100, y=30)

webdriver_path_button = tk.Button(window, text="Select Webdriver", command=get_webdriver_path, bg='white', fg='black',
                                  font=('Arial', 10, 'bold'), width=20)
webdriver_path_button.pack()
webdriver_path_button.place(x=430, y=30)

ip_address_entry = tk.Entry(window, width=50)
ip_address_entry.pack()
ip_address_entry.place(x=100, y=70)

ip_address_button = tk.Button(window, text="Enter IP address", command=lambda: [get_ip_address()], bg='white',
                              fg='black', font=('Arial', 10, 'bold'), width=20)
ip_address_button.pack()
ip_address_button.place(x=430, y=70)

submit_button = tk.Button(window, text="Submit", command=submit, bg='white', fg='black', font=('Arial', 10, 'bold'),
                          width=20)
submit_button.pack()
submit_button.place(x=100, y=110)

instructions_button = tk.Button(window, text="Instructions", command=show_instructions, bg='white', fg='black',
                                font=('Arial', 10, 'bold'), width=20)
instructions_button.pack()
instructions_button.place(x=430, y=110)

window.geometry("1000x600")
window.mainloop()