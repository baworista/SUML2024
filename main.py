import os
from sklearn.model_selection import train_test_split
from data_fetcher import download_and_extract_kaggle_data
from autoML import AutoGluonProcessor

# Define parameters
train_file_path = 'data/train.csv'
label_column = 'sii'

# Check if the CSV file exists, else download it
if not os.path.exists(train_file_path):
    print("Data file not found. Downloading now...")
    download_and_extract_kaggle_data(
        kaggle_username='aleksanderignacik',
        kaggle_key='1f9218f8383175887b7f88bd58132752',
        competition_name='child-mind-institute-problematic-internet-use',
        download_dir='./data'
    )
else:
    print("Data file already exists. Skipping download.")

# Instantiate AutoGluon processor
processor = AutoGluonProcessor(label_column=label_column)

# Preprocess entire dataset
full_df = processor.preprocess_data(train_file_path)

# Split into train and test sets
train_df, test_df = train_test_split(full_df, test_size=0.2, random_state=42)

# Train a model on the training set
predictor = processor.train_model(train_df, time_limit=60)

# Make predictions on the test set
predictions = processor.predict(predictor, test_df)
