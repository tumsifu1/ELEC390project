import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import h5py

data = pd.DataFrame(np.array(h5py.File('accelerometer_data.h5')['XZ']['walking']))

# don't have any missing data when I was testing
print(np.where(data[:] == '-'))
print((data[:] == '-').sum())

print(np.where(data.isna()))
print(data.isna().sum())

data.mask(data[:] == '-', other=np.nan, inplace=True)

data = data.astype('float64')

data.interpolate(method='linear', inplace=True)

# rolling window
window_size = 5
size5 = data.rolling(window_size).mean()

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(data[0], data[1])
ax.plot(data[0], size5[1])

plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(data[0], data[2])
ax.plot(data[0], size5[2])

plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(data[0], data[3])
ax.plot(data[0], size5[3])

plt.show()
