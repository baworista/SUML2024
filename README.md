
# AutoGluon Model Prediction App

This project demonstrates how to use the AutoGluon library to train and deploy machine learning models in a web application using **Streamlit**. The goal is to create an easy-to-use interface where users can input feature values in CSV format and get predictions based on a pre-trained model.

### Dataset Description
The Healthy Brain Network (HBN) dataset is a clinical sample of 5,000+ participants aged 5-22 years, containing various health and behavioral measures. The dataset is used to predict the Severity Impairment Index (sii), which indicates problematic internet use.

The dataset consists of several categories of information:

**Demographics**: Includes participant's age, sex, and enrollment season.  
**Children's Global Assessment Scale (CGAS)**: Measures the general functioning of youths, including season of participation and CGAS score. 
**Physical Measures**: Includes body mass index (BMI), height, weight, waist circumference, blood pressure, and heart rate.   
**FitnessGram**: Data related to cardiovascular fitness, including endurance stage, time, and various physical tests like push-ups, grip strength, and sit & reach.  
**Bio-electric Impedance Analysis**: Includes measures such as bone mineral content (BMC), body fat percentage, fat-free mass (FFM), and total body water.  
**Physical Activity Questionnaire (PAQ)**: Includes activity summary scores for adolescents and children.   
**Parent-Child Internet Addiction Test (PCIAT)**: Contains 20 questions assessing internet addiction behaviors, with a total score indicating severity (None, Mild, Moderate, Severe). 
**Sleep Disturbance Scale**: Measures sleep disturbance through total raw scores and T-scores.  
**Internet Use**: Measures the number of hours a participant uses the computer/internet per day.   

### Project Structure

The project consists of the following files:

1. **data_fetcher.py** – Downloads and extracts data from Kaggle competition.
2. **autoML.py** – Contains the `AutoGluonProcessor` class for loading, preprocessing data, training models, and making predictions using AutoGluon.
3. **main.py** – The main script that checks if the dataset is available, preprocesses the data, trains a model, and evaluates predictions.
4. **app.py** – The Streamlit application for the user interface where users can input data and get predictions from the trained model.

### Requirements

1. Python 3.7 or later
2. The following Python libraries:
   - `autogluon`
   - `sklearn`
   - `pandas`
   - `numpy`
   - `streamlit`
   - `io`
   - `concurrent.futures`
   - `zipfile`
   - `kaggle`

### Setup

1. Clone the repository or download the project files.
2. Install the required Python dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
   You will need to install the `autogluon` library for AutoML functionality.

3. **Kaggle Credentials**:
   - To download datasets from Kaggle, you need to create a Kaggle account and generate an API key (`kaggle.json`).
   - Save the `kaggle.json` file and place it in the directory specified by `kaggle_json_dir`. The script will automatically use it to authenticate.

4. **Directory Structure**:
   - Ensure the directory structure includes the following folders:
     - `./data/`: For storing the downloaded Kaggle dataset.
     - `saved_models/ag_model/`: For saving the trained AutoGluon model.

### How It Works

1. **Data Download and Extraction**: 
   - The `data_fetcher.py` script handles downloading the competition data from Kaggle and extracting it.
   - If the dataset is already downloaded, the script skips the download process.

2. **AutoGluon Model Training**:
   - The `autoML.py` script defines the `AutoGluonProcessor` class, which loads, preprocesses the dataset, and trains the model using AutoGluon.
   - The model is trained and saved in the directory `saved_models/ag_model/`.

3. **Streamlit Web Application**:
   - `app.py` is the Streamlit app that allows users to input feature values in CSV format. The input data is preprocessed and aligned with the model's expected features before making a prediction.
   - Predictions are made asynchronously to ensure a responsive user experience. If the prediction takes too long, the user will be notified with a timeout error.

### Running the Streamlit App

To run the app:

1. Navigate to the project directory in the terminal.
2. Run the following command to start the Streamlit web application:
   ```bash
   streamlit run app.py
   ```
3. The web app will open in your browser. You can now enter your feature values in CSV format and get predictions based on the trained model.

### Example Usage

Once the app is running, you'll see the following interface:

- Enter feature values in CSV format (headers are optional) in the provided text area. For example:

  ```
  Basic_Demos-Enroll_Season,Basic_Demos-Age, ...
  Fall,5, ...
  ```

- Click the **Make Prediction** button, and the model will process the input and return a prediction.

### Customizing the Model

If you'd like to change the model or the training data:

1. Modify the `label_column` in the `main.py` file to specify the column you want to predict.
2. Customize the dataset and model training process in `autoML.py` (for example, by adjusting training parameters or adding more preprocessing steps).

---

If you encounter any issues or have questions about the project, feel free to open an issue or contact the maintainer.
