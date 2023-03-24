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
axs = fig.add_subplot(111, projection='3d')

# Jumping data
x = member_jumping_data[:, 1]
y = member_jumping_data[:, 2]
z = member_jumping_data[:, 3]

# Create a grid of coordinates
X, Y = np.meshgrid(np.arange(x.min(), x.max(), 0.1), np.arange(y.min(), y.max(), 0.1))
Z = np.zeros_like(X)

# Fill in the grid with the corresponding values of z
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        idx = np.argmin((x - X[i,j])**2 + (y - Y[i,j])**2)
        Z[i,j] = z[idx]

# Create the surface plot
axs.plot_surface(X, Y, Z, cmap='viridis')

# Set the axis labels and title
axs.set_xlabel('X')
axs.set_ylabel('Y')
axs.set_zlabel('Z')
axs.set_title('AM Jumping Data (3D Heatmap)')

# Set the initial viewpoint
axs.view_init(elev=30, azim=120)

# Show the plot
plt.show()
