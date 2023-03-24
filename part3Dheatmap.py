import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data from HDF5 file
with h5py.File('./accelerometer_data.h5', 'r') as hdf:
    # Load walking and jumping data for specified member
    member_walking_data = hdf['MP/walking'][:]
    member_jumping_data = hdf['MP/jumping'][:]
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data from HDF5 file into a NumPy array
with h5py.File('./accelerometer_data.h5', 'r') as hdf:
    member_jumping_data = hdf['MP/jumping'][:]


# Jumping data
t_jump = member_jumping_data[:, 1] #Time
Y_jump = member_jumping_data[:, 2] #
Z_jump = member_jumping_data[:, 3]

# Walking data
t_walk = member_walking_data[:, 1] #Time
Y_walk = member_walking_data[:, 2] #
Z_walk = member_walking_data[:, 3]

#shift time values so that they are positive
t_jump = t_jump - t_jump.min()
t_walk = t_walk - t_walk.min()

# Create a 3D scatter plot
fig = plt.figure()
axs = fig.add_subplot(111, projection='3d')

# Create the scatter plot
axs.scatter(t_jump, Y_jump, Z_jump, label='Jumping Data', color='red')
axs.scatter(t_walk, Y_walk, Z_walk, label='Walking Data', color='blue')

# Set the axis labels and title
axs.set_xlabel('Time (s)')
axs.set_ylabel('X Acceleration (m/s^2)')
axs.set_zlabel('Y Acceleration (m/s^2)')
axs.set_title('MP Accelerometer Data (3D Scatter Plot)')

# Add legend
axs.legend()

# Show the plot
plt.show()
