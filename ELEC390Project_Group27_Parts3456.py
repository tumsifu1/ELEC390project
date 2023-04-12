# ELEC 390 Project Parts 3-6

import pandas as pd
import matplotlib.pyplot as plt
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from scipy.stats import skew
import joblib
import openpyxl


# Part 3 - Data visualization

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
    ax.plot(time, np.sqrt(acceleration_x ** 2 + acceleration_y ** 2 + acceleration_z ** 2), color=color, label=label)
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel('Absolute Acceleration')
    ax.legend()


# Read the data
jumping_data = read_data('jumping')
walking_data = read_data('walking')

# Time series plot of acceleration magnitude for walking and jumping
fig1, ax = plt.subplots(figsize=(12, 8))
plot_data(ax, walking_data, 'Absolute Acceleration vs Time', 'blue', 'Walking')
plot_data(ax, jumping_data, 'Absolute Acceleration vs Time', 'red', 'Jumping')
fig1.tight_layout()
plt.show()

# Histogram of acceleration magnitudes for walking and jumping
fig2, ax = plt.subplots(figsize=(12, 8))
ax.hist(np.linalg.norm(walking_data[:, 1:4], axis=1), bins=30, alpha=0.5, color='blue', label='Walking')
ax.hist(np.linalg.norm(jumping_data[:, 1:4], axis=1), bins=30, alpha=0.5, color='red', label='Jumping')
ax.set_title('Histogram of Absolute Acceleration')
ax.set_xlabel('Acceleration Magnitude')
ax.set_ylabel('Frequency')
ax.legend()
fig2.tight_layout()
plt.show()

# # Scatter plot of x vs y, x vs z, and y vs z for both walking and jumping data
fig3, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
axes[0].scatter(walking_data[:, 1], walking_data[:, 2], color='blue', label='Walking', alpha=0.5)
axes[0].scatter(jumping_data[:, 1], jumping_data[:, 2], color='red', label='Jumping', alpha=0.5)
axes[0].set_title('Acceleration Y vs Acceleration X')
axes[0].set_xlabel('Acceleration X')
axes[0].set_ylabel('Acceleration Y')
axes[0].legend()

axes[1].scatter(walking_data[:, 1], walking_data[:, 3], color='blue', label='Walking', alpha=0.5)
axes[1].scatter(jumping_data[:, 1], jumping_data[:, 3], color='red', label='Jumping', alpha=0.5)
axes[1].set_title('Acceleration Z vs Acceleration X')
axes[1].set_xlabel('Acceleration X')
axes[1].set_ylabel('Acceleration Z')
axes[1].legend()

axes[2].scatter(walking_data[:, 2], walking_data[:, 3], color='blue', label='Walking', alpha=0.5)
axes[2].scatter(jumping_data[:, 2], jumping_data[:, 3], color='red', label='Jumping', alpha=0.5)
axes[2].set_title('Acceleration Z vs Acceleration Y')
axes[2].set_xlabel('Acceleration Y')
axes[2].set_ylabel('Acceleration Z')
axes[2].legend()

fig3.tight_layout()
plt.show()

# #3D Scatter plot of x, y, and z accelerations for both walking and jumping data
fig4 = plt.figure(figsize=(12, 8))
ax = fig4.add_subplot(111, projection='3d')

ax.scatter(walking_data[:, 1], walking_data[:, 2], walking_data[:, 3], color='blue', label='Walking', alpha=0.5)
ax.scatter(jumping_data[:, 1], jumping_data[:, 2], jumping_data[:, 3], color='red', label='Jumping', alpha=0.5)

ax.set_title('3D Scatter Plot of Acceleration')
ax.set_xlabel('Acceleration X')
ax.set_ylabel('Acceleration Y')
ax.set_zlabel('Acceleration Z')
ax.legend()

plt.show()

# Plotting window data - read the training dataset
with h5py.File('./accelerometer_data.h5', 'r') as hdf:
    walking = hdf['dataset/train/walking'][:, :, 1:]
    jumping = hdf['dataset/train/jumping'][:, :, 1:]

# Plotting original Window for a specified index
window_index = 229

fig5, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(range(500), walking[window_index, :, 0], label='x acceleration')
ax1.plot(range(500), walking[window_index, :, 1], label='y acceleration')
ax1.plot(range(500), walking[window_index, :, 2], label='z acceleration')
ax1.set_title(f'Walking data for window {window_index+1}')
ax1.set_xlabel('Index')
ax1.set_ylabel('Acceleration (m/s^2)')
ax1.legend()

ax2.plot(range(500), jumping[window_index, :, 0], label='x acceleration')
ax2.plot(range(500), jumping[window_index, :, 1], label='y acceleration')
ax2.plot(range(500), jumping[window_index, :, 2], label='z acceleration')
ax2.set_title(f'Jumping data for window {window_index+1}')
ax2.set_xlabel('Index')
ax2.set_ylabel('Acceleration (m/s^2)')
ax2.legend()

plt.show()


# Moving average filter for visualisation
def MA_filter(data, wsize):
    df = pd.DataFrame(data)
    df_sma = df.rolling(wsize).mean()

    return df_sma


y_walking = walking[window_index, :, 1]
y_walking_sma5 = MA_filter(y_walking, 5)
y_walking_sma15 = MA_filter(y_walking, 15)

y_jumping = jumping[window_index, :, 1]
y_jumping_sma5 = MA_filter(y_jumping, 5)
y_jumping_sma15 = MA_filter(y_jumping, 15)

fig6, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(range(500), y_walking, label='Original')
ax1.plot(range(500), y_walking_sma5, label='SMA 5')
ax1.plot(range(500), y_walking_sma15, label='SMA 15')
ax1.set_title(f'Walking data for window {window_index+1}')
ax1.set_xlabel('Index')
ax1.set_ylabel('Acceleration (m/s^2)')
ax1.legend()

ax2.plot(range(500), y_jumping, label='Original')
ax2.plot(range(500), y_jumping_sma5, label='SMA 5')
ax2.plot(range(500), y_jumping_sma15, label='SMA 15')
ax2.set_title(f'Jumping data for window {window_index+1}')
ax2.set_xlabel('Index')
ax2.set_ylabel('Acceleration (m/s^2)')
ax2.legend()

plt.show()


# # Modified feature extraction function used for plotting purposes
def feature_extract(windows):
    features = np.zeros((windows.shape[0], 3, 4))
    for i in range(windows.shape[2]):
        for j in range(windows.shape[0]):
            window_data = windows[j, :, i]
            features[j, 0, i] = np.mean(window_data)
            features[j, 1, i] = np.std(window_data)
            features[j, 2, i] = np.var(window_data)
    return features


walking_feature = feature_extract(walking)
jumping_feature = feature_extract(jumping)

# Plotting histogram of standard deviation from windows
fig7, axs = plt.subplots(2, 4, figsize=(15, 10))

axs[0, 0].hist(walking_feature[:, 2, 0])
axs[0, 0].set_title('Walking X')
axs[0, 1].hist(walking_feature[:, 2, 1])
axs[0, 1].set_title('Walking Y')
axs[0, 2].hist(walking_feature[:, 2, 2])
axs[0, 2].set_title('Walking Z')
axs[0, 3].hist(walking_feature[:, 2, 3])
axs[0, 3].set_title('Walking Total')

axs[1, 0].hist(jumping_feature[:, 2, 0])
axs[1, 0].set_title('Jumping X')
axs[1, 1].hist(jumping_feature[:, 2, 1])
axs[1, 1].set_title('Jumping Y')
axs[1, 2].hist(jumping_feature[:, 2, 2])
axs[1, 2].set_title('Jumping Z')
axs[1, 3].hist(jumping_feature[:, 2, 3])
axs[1, 3].set_title('Jumping Total')

fig7.suptitle(f'Distribution of RMS for Walking and Jumping Window Data')
plt.show()

# Plot mean vs standard deviation for walking and jumping windows
fig8, ax = plt.subplots(figsize=(12, 8))
ax.scatter(walking_feature[:, 0, 3], walking_feature[:, 1, 3], color='blue', label='Walking', alpha=0.5)
ax.scatter(jumping_feature[:, 0, 3], jumping_feature[:, 1, 3], color='red', label='Jumping', alpha=0.5)
ax.set_title('Standard Deviation vs Mean of Absolute Acceleration (5-second windows)')
ax.set_xlabel('Mean')
ax.set_ylabel('Standard Deviation')
ax.legend()

fig8.tight_layout()
plt.show()

# Displaying metadata

# Create a new workbook and select the active worksheet
workbook = openpyxl.Workbook()
worksheet = workbook.active

# Data to be displayed in the table
data = {
    'Aaron': {
        'version': '1.1.11',
        'build': '10011',
        'fileFormat': '1.15',
        'deviceModel': 'iPhone14,3',
        'deviceBrand': 'Apple',
        'deviceBoard': '',
        'deviceManufacturer': '',
        'deviceBaseOS': '',
        'deviceCodename': '',
        'deviceRelease': '15.7.1',
        'depthFrontSensor': '1',
        'depthFrontResolution': '',
        'depthFrontRate': '',
        'depthBackSensor': '1',
        'depthBackResolution': '',
        'depthBackRate': ''
    },
    'Michael': {
        'version': '1.1.11',
        'build': '10011',
        'fileFormat': '1.15',
        'deviceModel': 'iPhone12,8',
        'deviceBrand': 'Apple',
        'deviceBoard': '',
        'deviceManufacturer': '',
        'deviceBaseOS': '',
        'deviceCodename': '',
        'deviceRelease': '16.1.1',
        'depthFrontSensor': '1',
        'depthFrontResolution': '',
        'depthFrontRate': '',
        'depthBackSensor': '0',
        'depthBackResolution': '',
        'depthBackRate': ''
    },
    'Xuchen': {
        'version': '1.1.11',
        'build': '1011102',
        'fileFormat': '1.15',
        'deviceModel': 'SM-A7070',
        'deviceBrand': 'samsung',
        'deviceBoard': 'sm6150',
        'deviceManufacturer': 'samsung',
        'deviceBaseOS': 'samsung/a70szc/a70s:11/RP1A.200720.012/A7070ZCU3CVD1:user/release-keys',
        'deviceCodename': 'REL',
        'deviceRelease': '11',
        'depthFrontSensor': '0',
        'depthFrontResolution': 'null',
        'depthFrontRate': 'null',
        'depthBackSensor': '0',
        'depthBackResolution': 'null',
        'depthBackRate': 'null'
    }
}

# Add the header row
header = ['Property', 'Aaron', 'Michael', 'Xuchen']
worksheet.append(header)

# Add the data rows
for key in data['Aaron'].keys():
    row = [key, data['Aaron'][key], data['Michael'].get(key, ''), data['Xuchen'].get(key, '')]
    worksheet.append(row)

# Autofit columns
for column_cells in worksheet.columns:
    length = max(len(str(cell.value)) for cell in column_cells)
    worksheet.column_dimensions[column_cells[0].column_letter].width = length

# Save the workbook
workbook.save('metadata.xlsx')

# ------------------------------------------------------------------------------- #

# Part 4 - Preprocessing
def data_processing(windows, w_size):
    # Create array for filtered data
    filtered_data = np.zeros((windows.shape[0], windows.shape[1]-w_size+1, windows.shape[2]))

    # Loop through each window and apply a moving average filter to each acceleration
    for i in range(windows.shape[0]):
        # Creating dataframes
        x_df = pd.DataFrame(windows[i, :, 0])
        y_df = pd.DataFrame(windows[i, :, 1])
        z_df = pd.DataFrame(windows[i, :, 2])
        total = windows[i, :, 3]  # MA filter not used on total acceleration

        # Apply MA filter
        x_sma = x_df.rolling(w_size).mean().values.ravel()
        y_sma = y_df.rolling(w_size).mean().values.ravel()
        z_sma = z_df.rolling(w_size).mean().values.ravel()

        # Discard the filtered NaN values
        x_sma = x_sma[w_size - 1:]
        y_sma = y_sma[w_size - 1:]
        z_sma = z_sma[w_size - 1:]
        total_sma = total[w_size - 1:]  # Keeping the same dimensions as other data

        # Store filtered data in array
        filtered_data[i, :, 0] = x_sma
        filtered_data[i, :, 1] = y_sma
        filtered_data[i, :, 2] = z_sma
        filtered_data[i, :, 3] = total_sma

    # Extract features (Part 5)
    feature_data = train_feature_extraction(filtered_data)

    # Create dataframes for further processing
    x_df = pd.DataFrame(feature_data[:, :, 0])
    y_df = pd.DataFrame(feature_data[:, :, 1])
    z_df = pd.DataFrame(feature_data[:, :, 2])
    total_df = pd.DataFrame(feature_data[:, :, 3])

    # Using z score to remove outliers in each dataframe
    for df in [x_df, y_df, z_df, total_df]:
        for i in range(df.shape[1]):
            column_data = df.iloc[:, i]
            z_scores = (column_data - column_data.mean())/column_data.std()
            column_data = column_data.mask(abs(z_scores) > 3, other=np.nan)  # Threshold at a z score of 3
            df.iloc[:, i] = column_data.fillna(filtered_data.mean())  # Fill NaN values with mean

    # Creating filtered feature array with labels for each measurement

    filtered_feature_data = np.concatenate((x_df, y_df, z_df, total_df), axis=0)
    labels = np.concatenate((np.ones((x_df.shape[0], 1)),
                             2 * np.ones((y_df.shape[0], 1)),
                             3 * np.ones((z_df.shape[0], 1)),
                             4 * np.ones((total_df.shape[0], 1))), axis=0)
    filtered_feature_data = np.hstack((filtered_feature_data, labels))

    return filtered_feature_data


# Part 5 - Feature extraction for training data
def train_feature_extraction(windows):
    # Create an empty array to hold the feature vectors
    features = np.zeros((windows.shape[0], 10, 4))

    # Iterate over each time window and extract the features
    for i in range(windows.shape[2]):
        for j in range(windows.shape[0]):
            # Extract the data from the window
            window_data = windows[j, :, i]

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
            features[j, :, i] = (max_val, min_val, range_val, mean_val, median_val, var_val, skew_val,
                                 rms_val, kurt_val, std_val)

    return features


# Part 5 - Feature extraction for test data
def test_feature_extraction(windows):
    # Create an empty array to hold the feature vectors
    features = np.zeros((windows.shape[0], 10, 4))

    # Iterate over each time window and extract the features
    for i in range(windows.shape[2]):
        for j in range(windows.shape[0]):
            # Extract the data from the window
            window_data = windows[j, :, i]

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
            features[j, :, i] = (max_val, min_val, range_val, mean_val, median_val, var_val, skew_val, rms_val,
                                 kurt_val, std_val)

    x_feature = features[:, :, 0]
    y_feature = features[:, :, 1]
    z_feature = features[:, :, 2]
    total_feature = features[:, :, 3]

    # Concatenate the feature arrays
    all_features = np.concatenate((x_feature, y_feature, z_feature, total_feature), axis=0)

    # Create a column of labels
    labels = np.concatenate((np.ones((x_feature.shape[0], 1)),
                             2 * np.ones((y_feature.shape[0], 1)),
                             3 * np.ones((z_feature.shape[0], 1)),
                             4 * np.ones((total_feature.shape[0], 1))), axis=0)

    # Add the labels column to the feature array
    all_features = np.hstack((all_features, labels))

    return all_features

# Part 6 - Classifier

# 6A - Data setup
# Import the data from the HDF5 file
with h5py.File('./accelerometer_data.h5', 'r') as hdf:
    train_walking_windows = hdf['dataset/train/walking'][:, :, 1:]
    train_jumping_windows = hdf['dataset/train/jumping'][:, :, 1:]
    test_walking_windows = hdf['dataset/test/walking'][:, :, 1:]
    test_jumping_windows = hdf['dataset/test/jumping'][:, :, 1:]

# Process data with a specified window size for the MA filter (Part 4 + Part 5)
window_size = 5
walking_filtered = data_processing(train_walking_windows, window_size)
jumping_filtered = data_processing(train_jumping_windows, window_size)

# Combine walking and jumping data into one dataset
training_features = np.concatenate((walking_filtered, jumping_filtered), axis=0)
training_labels = np.concatenate((np.zeros((walking_filtered.shape[0], 1)),
                                  np.ones((jumping_filtered.shape[0], 1))), axis=0)

# Extract features from test data and combine walking and jumping into dataset
test_walking_features = test_feature_extraction(test_walking_windows)
test_jumping_features = test_feature_extraction(test_jumping_windows)
test_features = np.concatenate((test_walking_features, test_jumping_features), axis=0)
test_labels = np.concatenate((np.zeros((test_walking_features.shape[0], 1)),
                              np.ones((test_jumping_features.shape[0], 1))), axis=0)

# Add labels to the train and test feature arrays
column_labels = np.array(
        ['max_val', 'min_val', 'range_val', 'mean_val', 'median_val', 'var_val', 'skew_val', 'rms_val', 'kurt_val',
         'std_val', 'measurement', 'activity'])
training_dataset = pd.DataFrame(np.hstack((training_features, training_labels)), columns=column_labels)
test_dataset = pd.DataFrame(np.hstack((test_features, test_labels)), columns=column_labels)

# Reading datasets and converting them into 4 separate dataframes
featureNumber = 10
xTrain = training_dataset[training_dataset.iloc[:, featureNumber] == 1]
yTrain = training_dataset[training_dataset.iloc[:, featureNumber] == 2]
zTrain = training_dataset[training_dataset.iloc[:, featureNumber] == 3]
allTrain = training_dataset[training_dataset.iloc[:, featureNumber] == 4]

xTest = test_dataset[test_dataset.iloc[:, featureNumber] == 1]
yTest = test_dataset[test_dataset.iloc[:, featureNumber] == 2]
zTest = test_dataset[test_dataset.iloc[:, featureNumber] == 3]
allTest = test_dataset[test_dataset.iloc[:, featureNumber] == 4]

# Part 6B - Creating the model

# Separating dataframes into train and test
X_xTrain = xTrain.iloc[:, 0:-2]  # Remove outer labels
X_xTest = xTest.iloc[:, 0:-2]

X_yTrain = yTrain.iloc[:, 0:-2]
X_yTest = yTest.iloc[:, 0:-2]

X_zTrain = zTrain.iloc[:, 0:-2]
X_zTest = zTest.iloc[:, 0:-2]

X_allTrain = allTrain.iloc[:, 0:-2]
X_allTest = allTest.iloc[:, 0:-2]

# Labels (same for all dataframes)
Y_combinedTrain = allTrain.iloc[:, -1]
Y_combinedTest = allTest.iloc[:, -1]

# Creating a combined train and test set with all the features for the 4 measurements
X_combinedTrain = np.zeros((X_xTrain.shape[0], 4*featureNumber))
X_combinedTest = np.zeros((X_xTest.shape[0], 4*featureNumber))
for k in range(featureNumber):
    X_combinedTrain[:, k] = X_xTrain.iloc[:, k]
    X_combinedTrain[:, k+featureNumber] = X_yTrain.iloc[:, k]
    X_combinedTrain[:, k+(2*featureNumber)] = X_zTrain.iloc[:, k]
    X_combinedTrain[:, k+(3*featureNumber)] = X_allTrain.iloc[:, k]
    X_combinedTest[:, k] = X_xTest.iloc[:, k]
    X_combinedTest[:, k + featureNumber] = X_yTest.iloc[:, k]
    X_combinedTest[:, k + (2 * featureNumber)] = X_zTest.iloc[:, k]
    X_combinedTest[:, k + (3 * featureNumber)] = X_allTest.iloc[:, k]

# Defining classifier and the pipeline with normalized inputs
l_reg = LogisticRegression(max_iter=10000)
clfCombined = make_pipeline(StandardScaler(), l_reg)

# Training the model and saving classifier as joblib file
clfCombined.fit(X_combinedTrain, Y_combinedTrain)
joblib.dump(clfCombined, 'classifier.joblib')

# Calculating the predictions and their probabilities
Y_predicted_combined = clfCombined.predict(X_combinedTest)
Y_clf_prob_combined = clfCombined.predict_proba(X_combinedTest)

# Plotting the confusion matrix for the model
cm = confusion_matrix(Y_combinedTest, Y_predicted_combined)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

# Plotting the ROC curve
fpr, tpr, _ = roc_curve(Y_combinedTest, Y_clf_prob_combined[:, 1], pos_label=clfCombined.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

# Outputting evaluation metrics
TP = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TN = cm[1, 1]

# Accuracy
accuracyCombined = accuracy_score(Y_combinedTest, Y_predicted_combined)
print('Accuracy of the model is: ', accuracyCombined)

# Recall
recallCombined = recall_score(Y_combinedTest, Y_predicted_combined)
print('Recall of the model is: ', recallCombined)

# Specificity
specificity = TN/(TN+FP)
print('Specificity of the model is: ', specificity)
# Calculating F1 score

# Precision
precision = TP/(TP+FP)
print('Precision of the model is: ', specificity)

# F1 Score
F1 = (2*TP)/(2*TP+FP+FN)
print('F1 Score of the model is: ', F1)

# Outputting the AUC
auc = roc_auc_score(Y_combinedTest, Y_clf_prob_combined[:, 1])
print('AUC of the model is: ', auc)
