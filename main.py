import os
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import data_fetcher

# Define paths to training and test CSV files
train_file_path = 'data/train.csv'
test_file_path = 'data/test.csv'

# Check if the CSV files already exist
if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
    # If either file does not exist, use the data_fetcher to download and extract the data
    print("Data files not found. Downloading now...")
    data_fetcher.download_and_extract_kaggle_data(
        kaggle_username='aleksanderignacik',
        kaggle_key='1f9218f8383175887b7f88bd58132752',
        competition_name='child-mind-institute-problematic-internet-use',
        download_dir='./data'
    )
else:
    print("Data files already exist. Skipping download.")

# Load the training and test datasets
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# Print basic information about the datasets
print(train_df.head())
print(train_df.info())

# Split the training data into features and target variable
X_train = train_df.iloc[:, 1:-2].values
y_train = train_df.iloc[:, -1].values

# Print the training feature matrix and target vector
print(X_train)
print(y_train)
