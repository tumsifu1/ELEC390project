import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data from HDF5 file
with h5py.File('./accelerometer_data.h5', 'r') as hdf:
    # Load walking and jumping data for specified member
    member_walking_data = hdf['AM/walking'][:]
    member_jumping_data = hdf['AM/jumping'][:]

# Plot acceleration vs. time for walking and jumping data
fig = plt.figure(figsize=(10, 10))
axs = fig.add_subplot(111, project = '3d')

# Walking data
x = member_jumping_data[:, 1]
y = member_jumping_data[:, 2]
z = member_jumping_data[:, 3]

heatmap, xedges, yedges = np.histogram2d(x, y, bins=20)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
axs.set_xlabel('X')
axs.set_ylabel('Y')
axs.set_zlabel('Z')

axs.set_title('XZ Jumping Data (3D Heatmap)')

axs.view_init(elev=30, azim=120)

axs.plot_surface(x, y, z, cmap='viridis')

plt.show()