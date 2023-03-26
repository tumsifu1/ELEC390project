import h5py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# Function to filter the data window by window
def data_filter(windows, wsize):
    filtered_data = np.zeros((windows.shape[0], windows.shape[1]-wsize+1, windows.shape[2]))

    for i in range(windows.shape[0]):
        # print("NaN values in iteration")
        # print("-----------------------------")

        x_df = pd.DataFrame(windows[i, :, 0])
        y_df = pd.DataFrame(windows[i, :, 1])
        z_df = pd.DataFrame(windows[i, :, 2])
        total_df = pd.DataFrame(windows[i, :, 3])

        # print(np.sum(x_df.isna()).sum())

        # Remove outliers using interquartile range (IQR) method for each axis
        for df in [x_df, y_df, z_df, total_df]:
            q1 = df.quantile(0.25)
            q3 = df.quantile(0.75)
            iqr = q3 - q1
            df[(df < (q1 - 1.5 * iqr)) | (df > (q3 + 1.5 * iqr))] = np.nan

        # Replace NaN values with the mean of the remaining values
        x_df.fillna(x_df.mean(), inplace=True)
        y_df.fillna(y_df.mean(), inplace=True)
        z_df.fillna(z_df.mean(), inplace=True)
        total_df.fillna(total_df.mean(), inplace=True)

        # print(np.sum(x_df.isna()).sum())

        x_sma = x_df.rolling(wsize).mean().values.ravel()
        y_sma = y_df.rolling(wsize).mean().values.ravel()
        z_sma = z_df.rolling(wsize).mean().values.ravel()
        total_sma = total_df.rolling(wsize).mean().values.ravel()

        # Discard the filtered NaN values
        x_sma = x_sma[wsize - 1:]
        y_sma = y_sma[wsize - 1:]
        z_sma = z_sma[wsize - 1:]
        total_sma = total_sma[wsize - 1:]

        # print(np.sum(np.isnan(x_sma)).sum())

        # Normalize the filtered data
        sc = StandardScaler()
        x_scaled = sc.fit_transform(x_sma.reshape(-1, 1)).ravel()
        y_scaled = sc.fit_transform(y_sma.reshape(-1, 1)).ravel()
        z_scaled = sc.fit_transform(z_sma.reshape(-1, 1)).ravel()
        total_scaled = sc.fit_transform(total_sma.reshape(-1, 1)).ravel()

        # print(np.sum(np.isnan(x_scaled)).sum())

        # Replace NaN values with linear interpolation
        x_clean = pd.Series(x_scaled).interpolate().values
        y_clean = pd.Series(y_scaled).interpolate().values
        z_clean = pd.Series(z_scaled).interpolate().values
        total_clean = pd.Series(total_scaled).interpolate().values

        # print(np.sum(np.isnan(x_clean)).sum())
        # print("-----------------------------")

        filtered_data[i, :, 0] = x_clean
        filtered_data[i, :, 1] = y_clean
        filtered_data[i, :, 2] = z_clean
        filtered_data[i, :, 3] = total_clean

    return filtered_data


# Function that allows data to be saved in a csv
def csv_merge(activity_data):
    # Define data type
    x = activity_data[:, :, 0]
    y = activity_data[:, :, 1]
    z = activity_data[:, :, 2]
    total = activity_data[:, :, 3]

    # Concatenate the data
    dataset = np.concatenate((x, y, z, total), axis=0)

    # Create a column of labels
    labels = np.concatenate((np.ones((x.shape[0], 1)),
                             2 * np.ones((y.shape[0], 1)),
                             3 * np.ones((z.shape[0], 1)),
                             4 * np.ones((total.shape[0], 1))), axis=0)

    # Add the labels column to the data
    dataset = np.hstack((dataset, labels))

    return dataset


# Read the dataset
with h5py.File('./accelerometer_data.h5', 'r') as hdf:
    walking = hdf['dataset/train/walking'][:, :, 1:]
    jumping = hdf['dataset/train/jumping'][:, :, 1:]

# Filter the data with a specified window size
window_size = 15
walking_filtered = data_filter(walking, window_size)
jumping_filtered = data_filter(jumping, window_size)

# Save the filtered data to a CSV file
walking_filtered_csv = csv_merge(walking_filtered)
jumping_filtered_csv = csv_merge(jumping_filtered)
np.savetxt('walkingfiltered.csv', jumping_filtered_csv, delimiter=',')
np.savetxt('jumpingfiltered.csv', jumping_filtered_csv, delimiter=',')

# Observe the filter; specify window index and activity (0=walking, 1=jumping)
window_index = 499
activity = 0
if activity == 0:
    prefilter = walking
    postfilter = walking_filtered
    activity_type = 'walking'
else:
    prefilter = jumping
    postfilter = jumping_filtered
    activity_type = 'jumping'

# Set up the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot the unfiltered data for specified window in the left subplot
ax1.plot(range(500), prefilter[window_index, :, 0], label='x acceleration')
ax1.plot(range(500), prefilter[window_index, :, 1], label='y acceleration')
ax1.plot(range(500), prefilter[window_index, :, 2], label='z acceleration')

# Set the title and axis labels for the left subplot
ax1.set_title(f'Unfiltered {activity_type} data for window {window_index+1}')
ax1.set_xlabel('Sample')
ax1.set_ylabel('Acceleration (m/s^2)')

# Add a legend for the left subplot
ax1.legend()

# Plot the filtered data for the window in the right subplot
ax2.plot(range(500-window_size+1), postfilter[window_index, :, 0], label='x acceleration')
ax2.plot(range(500-window_size+1), postfilter[window_index, :, 1], label='y acceleration')
ax2.plot(range(500-window_size+1), postfilter[window_index, :, 2], label='z acceleration')

# Set the title and axis labels for the right subplot
ax2.set_title(f'Filtered {activity_type} data for window {window_index+1} using a window size of {window_size}')
ax2.set_xlabel('Sample')
ax2.set_ylabel('Acceleration (m/s^2)')

# Add a legend for the right subplot
ax2.legend()

# Show the plot
plt.show()
