
#ELEC 390 Part 3 - Data visulation 
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_data(activity):
    with h5py.File('./accelerometer_data.h5', 'r') as f:
        member_walking_data = f['MP/walking'][:]
        member_jumping_data = f['MP/jumping'][:]
    if activity == 'walking':
        return member_walking_data
    elif activity == 'jumping':
        return member_jumping_data
    else:
        raise ValueError('Invalid activity value')

def plot_data(ax, data, title, color, label):
    time = data[:, 0]
    acceleration_x = data[:, 1]
    acceleration_y = data[:, 2]
    acceleration_z = data[:, 3]
    ax.plot(time, np.sqrt(acceleration_x**2 + acceleration_y**2 + acceleration_z**2), color=color, label=label)
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Acceleration Magnitude')
    ax.legend()

# Read the data
jumping_data = read_data('jumping')
walking_data = read_data('walking')

# # Time series plot of acceleration magnitude for walking and jumping
# fig, ax = plt.subplots(figsize=(12, 8))
# plot_data(ax, walking_data, 'Acceleration Magnitude vs Time', 'blue', 'Walking')
# plot_data(ax, jumping_data, 'Acceleration Magnitude vs Time', 'red', 'Jumping')
# fig.tight_layout()
# plt.show()

# # Histogram of acceleration magnitudes for walking and jumping
# fig, ax = plt.subplots(figsize=(12, 8))
# ax.hist(np.linalg.norm(walking_data[:, 1:4], axis=1), bins=30, alpha=0.5, color='blue', label='Walking')
# ax.hist(np.linalg.norm(jumping_data[:, 1:4], axis=1), bins=30, alpha=0.5, color='red', label='Jumping')
# ax.set_title('Histogram of Acceleration Magnitudes')
# ax.set_xlabel('Acceleration Magnitude')
# ax.set_ylabel('Frequency')
# ax.legend()
# fig.tight_layout()
# plt.show()

# # Scatter plot of x vs y, x vs z, and y vs z for both walking and jumping data
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
# axes[0].scatter(walking_data[:, 1], walking_data[:, 2], color='blue', label='Walking', alpha=0.5)
# axes[0].scatter(jumping_data[:, 1], jumping_data[:, 2], color='red', label='Jumping', alpha=0.5)
# axes[0].set_title('Acceleration Y vs Acceleration X')
# axes[0].set_xlabel('Acceleration X')
# axes[0].set_ylabel('Acceleration Y')
# axes[0].legend()

# axes[1].scatter(walking_data[:, 1], walking_data[:, 3], color='blue', label='Walking', alpha=0.5)
# axes[1].scatter(jumping_data[:, 1], jumping_data[:, 3], color='red', label='Jumping', alpha=0.5)
# axes[1].set_title('Acceleration Z vs Acceleration X')
# axes[1].set_xlabel('Acceleration X')
# axes[1].set_ylabel('Acceleration Z')
# axes[1].legend()

# axes[2].scatter(walking_data[:, 2], walking_data[:, 3], color='blue', label='Walking', alpha=0.5)
# axes[2].scatter(jumping_data[:, 2], jumping_data[:, 3], color='red', label='Jumping', alpha=0.5)
# axes[2].set_title('Acceleration Z vs Acceleration Y')
# axes[2].set_xlabel('Acceleration Y')
# axes[2].set_ylabel('Acceleration Z')
# axes[2].legend()

# fig.tight_layout()
# plt.show()

# #3D Scatter plot of x, y, and z accelerations for both walking and jumping dat
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(walking_data[:, 1], walking_data[:, 2], walking_data[:, 3], color='blue', label='Walking', alpha=0.5)
# ax.scatter(jumping_data[:, 1], jumping_data[:, 2], jumping_data[:, 3], color='red', label='Jumping', alpha=0.5)

# ax.set_title('3D Scatter Plot of Acceleration')
# ax.set_xlabel('Acceleration X')
# ax.set_ylabel('Acceleration Y')
# ax.set_zlabel('Acceleration Z')
# ax.legend()

# plt.show()

#plotting windows
def plot_window(walking_data, jumping_data, window_start, window_size):
    fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot walking data window
    walking_window = walking_data[window_start:window_start + window_size, :]
    plot_data(ax[0], walking_window, 'Walking: Acceleration Magnitude vs Time', 'blue', 'Walking')

    # Plot jumping data window
    jumping_window = jumping_data[window_start:window_start + window_size, :]
    plot_data(ax[1], jumping_window, 'Jumping: Acceleration Magnitude vs Time', 'red', 'Jumping')

    fig.tight_layout()
    plt.show()

window_start = 1000
window_size = 500
plot_window(walking_data, jumping_data, window_start, window_size)

def window_summary_statistics(data, window_size=500, step_size=100):
    num_segments = (len(data) - window_size) // step_size + 1
    segments = [data[i * step_size : i * step_size + window_size] for i in range(num_segments)]
    
    means = []
    stds = []
    
    for segment in segments:
        accel_mag = np.linalg.norm(segment[:, 1:4], axis=1)
        means.append(np.mean(accel_mag))
        stds.append(np.std(accel_mag))
    
    return means, stds

# Calculate mean and standard deviation for walking and jumping data
walking_means, walking_stds = window_summary_statistics(walking_data)
jumping_means, jumping_stds = window_summary_statistics(jumping_data)

# Plot mean vs standard deviation for walking and jumping windows
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(walking_means, walking_stds, color='blue', label='Walking', alpha=0.5)
ax.scatter(jumping_means, jumping_stds, color='red', label='Jumping', alpha=0.5)
ax.set_title('Standard Deviation vs Mean of Acceleration Magnitude (5-second windows)')
ax.set_xlabel('Mean Acceleration Magnitude')
ax.set_ylabel('Standard Deviation of Acceleration Magnitude')
ax.legend()

fig.tight_layout()
plt.show()

