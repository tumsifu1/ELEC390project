import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load data from HDF5 file
with h5py.File('./accelerometer_data.h5', 'r') as hdf:
    # Load walking and jumping data for specified member
    member_walking_data = hdf['AM/walking'][:]
    member_jumping_data = hdf['AM/jumping'][:]

# Plot acceleration vs. time for walking and jumping data
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Walking data
axZero = np.arange(0, len(member_walking_data)/100, 0.01)
axs[0].plot(axZero, member_walking_data[:, 1], label='x')
axs[0].plot(axZero, member_walking_data[:, 2], label='y')
axs[0].plot(axZero, member_walking_data[:, 3], label='z')
axs[0].set_title('XZ Walking Data')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Acceleration (m/s^2)')
axs[0].legend()

# Jumping data
axOne = np.arange(0, len(member_jumping_data)/100, 0.01)
axs[1].plot(axOne, member_jumping_data[:, 1], label='x')
axs[1].plot(axOne, member_jumping_data[:, 2], label='y')
axs[1].plot(axOne, member_jumping_data[:, 3], label='z')
axs[1].set_title('XZ Jumping Data')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Acceleration (m/s^2)')
axs[1].legend()

plt.show()