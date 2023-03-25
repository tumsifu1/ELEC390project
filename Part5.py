import h5py
import numpy as np
from scipy.stats import skew


def extract_features(windows):
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
            features[j, :, i] = (max_val, min_val, range_val, mean_val, median_val, var_val, skew_val, rms_val, kurt_val, std_val)

    xfeature = features[:, :, 0]
    yfeature = features[:, :, 1]
    zfeature = features[:, :, 2]
    allfeature = features[:, :, 3]

    # Concatenate the feature arrays
    all_features = np.concatenate((xfeature, yfeature, zfeature, allfeature), axis=0)

    # Create a column of labels
    labels = np.concatenate((np.ones((xfeature.shape[0], 1)),
                             2 * np.ones((yfeature.shape[0], 1)),
                             3 * np.ones((zfeature.shape[0], 1)),
                             4 * np.ones((allfeature.shape[0], 1))), axis=0)

    # Add the labels column to the feature array
    all_features = np.hstack((all_features, labels))

    return all_features


# Read the datasets and extract features
with h5py.File('./accelerometer_data.h5', 'r') as hdf:
    # Save the walking features to a CSV file
    walking_windows = hdf['dataset/train/walking'][:, :, 1:]
    walking_features = extract_features(walking_windows)
    np.savetxt('walkingfeatures.csv', walking_features, delimiter=',')

    # Save the jumping features to a CSV file
    jumping_windows = hdf['dataset/train/jumping'][:, :, 1:]
    jumping_features = extract_features(jumping_windows)
    np.savetxt('jumpingfeatures.csv', jumping_features, delimiter=',')

    # Save the test features to a CSV file

    test_windows = hdf['dataset/test/data'][:, :, 1:]
    test_features = extract_features(test_windows)
    np.savetxt('testfeatures.csv', test_features, delimiter=',')