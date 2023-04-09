# ELEC 390 Project Part 2 - Creating HDF5 File for Data

# Import Libraries
import h5py
import numpy as np
import pandas as pd

# Load accelerometer data for each member from CSV files
# Each member has walking and jumping data, with four different types of collection
# The desired sampling rate is 100 Hz, and each type of collection is limited to 115 seconds
# 115 seconds for each collection with 100 samples per second
dataRange = 11500

# Member 1: MP

# Adjust Data to 100 Hz with a factor
M1DataCheck = pd.read_csv('UserData/MPWalkingBackPocket.csv')
M1SamplingFactor = round(len(M1DataCheck)/(100*M1DataCheck.iloc[-1, 0]))

# Walking Data
member1_walking_data = pd.concat([
    pd.read_csv('UserData/MPWalkingBackPocket.csv', nrows=M1SamplingFactor*dataRange), pd.read_csv('UserData/MPWalkingFrontPocket.csv', nrows=M1SamplingFactor*dataRange),
    pd.read_csv('UserData/MPWalkingJacketPocket.csv', nrows=dataRange), pd.read_csv('UserData/MPWalkingHand.csv', nrows=dataRange)
], ignore_index=True)

# Jumping Data
member1_jumping_data = pd.concat([
    pd.read_csv('UserData/MPJumpingBackPocket.csv', nrows=M1SamplingFactor*dataRange), pd.read_csv('UserData/MPJumpingFrontPocket.csv', nrows=M1SamplingFactor*dataRange),
    pd.read_csv('UserData/MPJumpingJacketPocket.csv', nrows=M1SamplingFactor*dataRange), pd.read_csv('UserData/MPJumpingHand.csv', nrows=M1SamplingFactor*dataRange)
], ignore_index=True)

if M1SamplingFactor > 1:
    member1_jumping_data = member1_jumping_data.drop(member1_jumping_data.index[1::M1SamplingFactor])
    member1_walking_data = member1_walking_data.drop(member1_walking_data.index[1::M1SamplingFactor])

# Member 2: AM
M2DataCheck = pd.read_csv('UserData/AMWalkingBackPocket.csv')
M2SamplingFactor = round(len(M2DataCheck)/(100*M2DataCheck.iloc[-1, 0]))

member2_walking_data = pd.concat([
    pd.read_csv('UserData/AMWalkingBackPocket.csv', nrows=M2SamplingFactor*dataRange), pd.read_csv('UserData/AMWalkingFrontPocket.csv', nrows=M2SamplingFactor*dataRange),
    pd.read_csv('UserData/AMWalkingJacketPocket.csv', nrows=M2SamplingFactor*dataRange), pd.read_csv('UserData/AMWalkingHand.csv', nrows=M2SamplingFactor*dataRange)
], ignore_index=True)

member2_jumping_data = pd.concat([
    pd.read_csv('UserData/AMJumpingBackPocket.csv', nrows=M2SamplingFactor*dataRange), pd.read_csv('UserData/AMJumpingFrontPocket.csv', nrows=M2SamplingFactor*dataRange),
    pd.read_csv('UserData/AMJumpingJacketPocket.csv', nrows=M2SamplingFactor*dataRange), pd.read_csv('UserData/AMJumpingHand.csv', nrows=M2SamplingFactor*dataRange)
], ignore_index=True)

if M2SamplingFactor > 1:
    member2_jumping_data = member2_jumping_data.drop(member2_jumping_data.index[1::M2SamplingFactor])
    member2_walking_data = member2_walking_data.drop(member2_walking_data.index[1::M2SamplingFactor])

# Member 3: XZ
M3DataCheck = pd.read_csv('UserData/XZWalkingBackPocket.csv')
M3SamplingFactor = round(len(M3DataCheck)/(100*M3DataCheck.iloc[-1, 0]))

member3_walking_data = pd.concat([
    pd.read_csv('UserData/XZWalkingBackPocket.csv', nrows=M3SamplingFactor*dataRange), pd.read_csv('UserData/XZWalkingFrontPocket.csv', nrows=M3SamplingFactor*dataRange),
    pd.read_csv('UserData/XZWalkingJacketPocket.csv', nrows=M3SamplingFactor*dataRange), pd.read_csv('UserData/XZWalkingHand.csv', nrows=M3SamplingFactor*dataRange)
], ignore_index=True)


member3_jumping_data = pd.concat([
    pd.read_csv('UserData/XZJumpingBackPocket.csv', nrows=2*dataRange), pd.read_csv('UserData/XZJumpingFrontPocket.csv', nrows=2*dataRange),
    pd.read_csv('UserData/XZJumpingJacketPocket.csv', nrows=2*dataRange), pd.read_csv('UserData/XZJumpingHand.csv', nrows=2*dataRange)
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

# Combining data
walking_combined_data = pd.concat([
    member1_walking_data, member2_walking_data, member3_walking_data
], ignore_index=True)
jumping_combined_data = pd.concat([
    member1_jumping_data, member2_jumping_data, member3_jumping_data
], ignore_index=True)
combined_data = pd.concat([walking_combined_data, jumping_combined_data], ignore_index=True)

walking_combined_data = walking_combined_data.to_numpy()
np.random.shuffle(walking_combined_data)
walking_combined_df = pd.DataFrame(walking_combined_data)
walking_combined_df.to_csv('walking_combined_data.csv', index=False)

jumping_combined_data = jumping_combined_data.to_numpy()
np.random.shuffle(jumping_combined_data)
jumping_combined_df = pd.DataFrame(jumping_combined_data)
jumping_combined_df.to_csv('jumping_combined_data.csv', index=False)

combined_data = combined_data.to_numpy()
np.random.shuffle(combined_data)
num_test = int(0.1 * len(combined_data))
test_combined_data = combined_data[:num_test]
test_combined_df = pd.DataFrame(test_combined_data)
test_combined_df.to_csv('test_combined_data.csv', index=False)

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

            # Segment the data into 5-second windows
            num_segments = (len(data) - 500) // 100 + 1
            segments = [data[(i * 100):(i * 100 + 500)] for i in range(num_segments)]

            # Label the segments with the member name and activity
            labels = [f'{member_name}_{activity}' for _ in range(num_segments)]

            # Combine the segments and labels for this position/activity combination
            all_segments.extend(list(zip(segments, labels)))

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
    test_group.create_dataset('walking', data=[seg[0] for seg in test_segments if 'walking' in seg[1]])
    test_group.create_dataset('jumping', data=[seg[0] for seg in test_segments if 'jumping' in seg[1]])

#