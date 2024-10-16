import os
import pandas as pd

# Define the path to the directory containing Parquet files
parquet_dir = 'data/series_train.parquet'

# List all the Parquet files in the directory
parquet_files = os.listdir(parquet_dir)

# Initialize an empty DataFrame to store the aggregated features for all participants
aggregated_data = pd.DataFrame()

# Loop through each file in the Parquet directory
for file_name in parquet_files:
    # Check if the file name matches the expected format (e.g., 'id=12345')
    if file_name.startswith('id='):
        # Extract the ID from the file name
        participant_id = file_name.split('=')[1]

        # Create the full file path
        file_path = os.path.join(parquet_dir, file_name)

        # Load the Parquet file for the current participant
        time_series_data = pd.read_parquet(file_path)
        print(len(time_series_data))

        # Convert 'time_of_day' to datetime (if needed)
        time_series_data['time_of_day'] = pd.to_datetime(time_series_data['time_of_day'], format='%H:%M:%S.%f')
        time_series_data.set_index('time_of_day', inplace=True)

        # Aggregate the time-series data (e.g., 5-minute windows)
        aggregated = time_series_data.resample('5T').agg({
            'X': ['mean', 'std', 'max'],
            'Y': ['mean', 'std', 'max'],
            'Z': ['mean', 'std', 'max'],
            'enmo': ['mean', 'std'],
            'anglez': ['mean'],
            'light': ['mean', 'max']
        })

        # Flatten the multi-level column names
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]

        # Add the participant ID as a column
        aggregated['id'] = participant_id

        # Append the aggregated data to the main DataFrame
        aggregated_data = pd.concat([aggregated_data, aggregated], axis=0)

# Inspect the aggregated data for all participants
print(aggregated_data.head())
