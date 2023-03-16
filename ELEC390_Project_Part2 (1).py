# ELEC 390 Project Part 2 - Creating HDF5 File for Data

# Import Libraries
import h5py
import numpy as np
import pandas as pd

# Load accelerometer data for each member from CSV files
# Each member has walking and jumping data, with four different types of collection
# The desired sampling rate is 100 Hz, and each type of collection is limited to 115 seconds
# Ignore index command is used to avoid data overlapping issues with csv files

# 115 seconds for each collection with 100 samples per second
dataRange = 11500

# Member 1: MP

# Adjust Data to 100 Hz with a factor
M1DataCheck = pd.read_csv('MPWalkingBackPocket.csv')
M1SamplingFactor = round(len(M1DataCheck)/(100*M1DataCheck.iloc[-1, 0]))

# Walking Data
member1_walking_data = pd.concat([
    pd.read_csv('MPWalkingBackPocket.csv', nrows=M1SamplingFactor*dataRange), pd.read_csv('MPWalkingFrontPocket.csv', nrows=M1SamplingFactor*dataRange),
    pd.read_csv('MPWalkingJacketPocket.csv', nrows=dataRange), pd.read_csv('MPWalkingHand.csv', nrows=dataRange)
], ignore_index=True)

# Jumping Data
member1_jumping_data = pd.concat([
    pd.read_csv('MPJumpingBackPocket.csv', nrows=M1SamplingFactor*dataRange), pd.read_csv('MPJumpingFrontPocket.csv', nrows=M1SamplingFactor*dataRange),
    pd.read_csv('MPJumpingJacketPocket.csv', nrows=M1SamplingFactor*dataRange), pd.read_csv('MPJumpingHand.csv', nrows=M1SamplingFactor*dataRange)
], ignore_index=True)

if M1SamplingFactor > 1:
    member1_jumping_data = member1_jumping_data.drop(member1_jumping_data.index[1::M1SamplingFactor])
    member1_walking_data = member1_walking_data.drop(member1_walking_data.index[1::M1SamplingFactor])

# Member 2: AM
M2DataCheck = pd.read_csv('AMWalkingBackPocket.csv')
M2SamplingFactor = round(len(M2DataCheck)/(100*M2DataCheck.iloc[-1, 0]))

member2_walking_data = pd.concat([
    pd.read_csv('AMWalkingBackPocket.csv', nrows=M2SamplingFactor*dataRange), pd.read_csv('AMWalkingFrontPocket.csv', nrows=M2SamplingFactor*dataRange),
    pd.read_csv('AMWalkingJacketPocket.csv', nrows=M2SamplingFactor*dataRange), pd.read_csv('AMWalkingHand.csv', nrows=M2SamplingFactor*dataRange)
], ignore_index=True)

member2_jumping_data = pd.concat([
    pd.read_csv('AMJumpingBackPocket.csv', nrows=M2SamplingFactor*dataRange), pd.read_csv('AMJumpingFrontPocket.csv', nrows=M2SamplingFactor*dataRange),
    pd.read_csv('AMJumpingJacketPocket.csv', nrows=M2SamplingFactor*dataRange), pd.read_csv('AMJumpingHand.csv', nrows=M2SamplingFactor*dataRange)
], ignore_index=True)

if M2SamplingFactor > 1:
    member2_jumping_data = member2_jumping_data.drop(member2_jumping_data.index[1::M2SamplingFactor])
    member2_walking_data = member2_walking_data.drop(member2_walking_data.index[1::M2SamplingFactor])

# Member 3: XZ
M3DataCheck = pd.read_csv('XZWalkingBackPocket.csv')
M3SamplingFactor = round(len(M3DataCheck)/(100*M3DataCheck.iloc[-1, 0]))

member3_walking_data = pd.concat([
    pd.read_csv('XZWalkingBackPocket.csv', nrows=M3SamplingFactor*dataRange), pd.read_csv('XZWalkingFrontPocket.csv', nrows=M3SamplingFactor*dataRange),
    pd.read_csv('XZWalkingJacketPocket.csv', nrows=M3SamplingFactor*dataRange), pd.read_csv('XZWalkingHand.csv', nrows=M3SamplingFactor*dataRange)
], ignore_index=True)


member3_jumping_data = pd.concat([
    pd.read_csv('XZJumpingBackPocket.csv', nrows=2*dataRange), pd.read_csv('XZJumpingFrontPocket.csv', nrows=2*dataRange),
    pd.read_csv('XZJumpingJacketPocket.csv', nrows=2*dataRange), pd.read_csv('XZJumpingHand.csv', nrows=2*dataRange)
], ignore_index=True)

if M3SamplingFactor > 1:
    member3_jumping_data = member3_jumping_data.drop(member3_jumping_data.index[1::M3SamplingFactor])
    member3_walking_data = member3_walking_data.drop(member3_walking_data.index[1::M3SamplingFactor])

# Combine all member data into a dictionary
all_data = {
    'MP': {'walking': member1_walking_data, 'jumping': member1_jumping_data},
    'AM': {'walking': member2_walking_data, 'jumping': member2_jumping_data},
    'XZ': {'walking': member3_walking_data, 'jumping': member3_jumping_data}
}

# Create an HDF5 file with structure:
# Top Level; Dataset[Train(Walk, Jump), Test(Walk, Jump)], MP(Walk, Jump), AM(Walk, Jump), XZ(Walk, Jump)
with h5py.File('./accelerometer_data.h5', 'w') as hdf:
    # Create sub groups for each member
    for member_name, member_data in all_data.items():
        member_group = hdf.create_group(member_name)
        member_group.create_dataset('walking', data=member_data['walking'])
        member_group.create_dataset('jumping', data=member_data['jumping'])

    # Create a sub group for the dataset
    dataset_group = hdf.create_group('dataset')

    # Segment and shuffle all accelerometer data
    all_segments = []
    for member_name, member_data in all_data.items():
        for activity in ['walking', 'jumping']:
            data = member_data[activity]

            # Segment the data into 5-second windows']:
            windows = [data[(i * 100):(i * 100 + 500)] for i in range(len(data) // 100 - 4)]

            # Label the segments with the member name and activity
            labels = [f'{member_name}_{activity}' for _ in range(len(windows))]

            # Combine the segments and labels for this position/activity combination
            all_segments.extend(list(zip(windows, labels)))

    # Shuffle the segmented data
    np.random.shuffle(all_segments)

    # Split the segmented data into 90% train and 10% test
    num_train = int(0.9 * len(all_segments))
    train_segments = all_segments[:num_train]
    test_segments = all_segments[num_train:]

    # Create sub groups for train and test datasets
    train_group = dataset_group.create_group('train')
    test_group = dataset_group.create_group('test')

    # Add walking and jumping datasets to train and test sub-groups
    train_group.create_dataset('walking', data=[seg[0] for seg in train_segments if 'walking' in seg[1]])
    train_group.create_dataset('jumping', data=[seg[0] for seg in train_segments if 'jumping' in seg[1]])
    test_group.create_dataset('data', data=[seg[0] for seg in test_segments])