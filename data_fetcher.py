import os
import zipfile


# Set the KAGGLE_CONFIG_DIR environment variable to specify where the kaggle.json is located
def set_kaggle_credentials(kaggle_json_dir):
    # Set the environment variable for the Kaggle API to find the credentials
    os.environ['KAGGLE_CONFIG_DIR'] = kaggle_json_dir


# Function to download and extract data
def download_and_extract_kaggle_data(kaggle_username, kaggle_key, competition_name, download_dir='./data',
                                     kaggle_json_dir=None):
    # If kaggle_json_dir is not provided, set it to the default project folder
    if kaggle_json_dir is None:
        kaggle_json_dir = os.path.join(os.getcwd(), '.kaggle')  # Default to the current project folder

    # Ensure the .kaggle folder exists
    os.makedirs(kaggle_json_dir, exist_ok=True)

    # Define the path to the kaggle.json file
    kaggle_json_path = os.path.join(kaggle_json_dir, 'kaggle.json')

    # Create the kaggle.json file if it doesn't exist
    if not os.path.exists(kaggle_json_path):
        with open(kaggle_json_path, 'w') as f:
            f.write(f'{{"username": "{kaggle_username}", "key": "{kaggle_key}"}}')

    # Set the environment variable to point to the custom .kaggle folder
    set_kaggle_credentials(kaggle_json_dir)

    # Import KaggleApi after setting the environment variable
    from kaggle.api.kaggle_api_extended import KaggleApi

    # Initialize Kaggle API and authenticate
    api = KaggleApi()
    api.authenticate()

    # Ensure the directory for data exists
    os.makedirs(download_dir, exist_ok=True)

    # Download competition files
    print("Downloading data...")
    api.competition_download_files(competition_name, path=download_dir)

    # Unzip the files if they are in a zip format
    zip_file_path = os.path.join(download_dir, f'{competition_name}.zip')
    if os.path.exists(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(download_dir)
        print("Data downloaded and extracted!")
    else:
        print(f"Zip file not found at {zip_file_path}")



