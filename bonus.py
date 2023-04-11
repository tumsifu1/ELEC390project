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
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox

#global var
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
    window_size = 50
    iterations = 15
    time.sleep(5)
    for n in range(iterations):
        # Create an empty array with shape (1, 100, 4)
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

                # Collect data from elements 6, 7, 8, 9
                soup = BeautifulSoup(driver.page_source, 'html.parser')

                value_x = float(soup.find('div', {'id': 'element6'}).find('span', {'class': 'valueNumber'}).text.strip())
                value_y = float(soup.find('div', {'id': 'element7'}).find('span', {'class': 'valueNumber'}).text.strip())
                value_z = float(soup.find('div', {'id': 'element8'}).find('span', {'class': 'valueNumber'}).text.strip())
                value_total = np.sqrt(value_x ** 2 + value_y ** 2 + value_z ** 2)

                # Convert values to floats and store in data_array
                x_data_array[index] = value_x
                y_data_array[index] = value_y
                z_data_array[index] = value_z
                total_data_array[index] = value_total

                index += 1

                # Break loop after 30 data points
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
        print(Y_predicted)
        window.after(0, update_activity)
        
        #change_window_color(Y_predicted[0])
        
#driver = webdriver.Chrome('C:\\Users\\mjpat\\Downloads\\chromedriver_win32\\chromedriver.exe')
#driver.get('http://192.168.2.24/')

#functions for UI
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
    global driver, webdriver_path, ip_address
    if webdriver_path and ip_address:
        driver = webdriver.Chrome(webdriver_path)
        driver.get(f'http://{ip_address}/')
        process_data_and_predict()
    else:
        messagebox.showwarning("Warning", "Please select a web driver and enter a valid IP address before submitting.")

def show_instructions():
    instructions = """1. Open the Phyphox app
2. Click on "Acceleration without G"
3. Activate the "Access with distance" option by clicking the three buttons in the top right corner of the Phyphox interface
4. Input the URL provided by Phyphox into the UI text box and go to it on Chrome
5. Input the location of the web driver into the UI text box"""

    messagebox.showinfo("Instructions", instructions)


def update_activity(activity):
    global predicted_activity
    if activity == 1:
        activity_icon_label.config(image=activity_icon_walking)
    else:
        activity_icon_label.config(image=activity_icon_jumping)

#UI
window = tk.Tk()
activity_icon_walking = tk.PhotoImage(file='walking.png')
activity_icon_jumping = tk.PhotoImage(file='jumping.png')
window.configure(bg='black')
window.title('Acceleration Classifier')

webdriver_path_label = tk.Label(window, width=40, text="", anchor='w', bg='white', relief='sunken')
webdriver_path_label.pack()
webdriver_path_label.place(x=100, y=30)

webdriver_path_button = tk.Button(window, text="Select Webdriver", command=get_webdriver_path, bg='white', fg='black', font=('Arial', 10, 'bold'), width=20)
webdriver_path_button.pack()
webdriver_path_button.place(x=430, y=30)

ip_address_entry = tk.Entry(window, width=50)
ip_address_entry.pack()
ip_address_entry.place(x=100, y=70)

ip_address_button = tk.Button(window, text="Enter IP Address", command=lambda: [get_ip_address()], bg='white', fg='black', font=('Arial', 10, 'bold'), width=20)
ip_address_button.pack()
ip_address_button.place(x=430, y=70)

submit_button = tk.Button(window, text="Submit", command=submit, bg='white', fg='black', font=('Arial', 10, 'bold'), width=20)
submit_button.pack()
submit_button.place(x=100, y=110)

instructions_button = tk.Button(window, text="Instructions", command=show_instructions, bg='white', fg='black', font=('Arial', 10, 'bold'), width=20)
instructions_button.pack()
instructions_button.place(x=430, y=110)

activity_icon_label = tk.Label(window, bg='white', image='', width=100, height=50)
activity_icon_label.pack()
activity_icon_label.place(x=100, y=150)



window.geometry("1000x600")
window.mainloop()