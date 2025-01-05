import os
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split

import data_fetcher
from autogluon.tabular import TabularDataset, TabularPredictor


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

# Print basic information about the datasets
print(train_df.head())
print(train_df.info())

label_column = 'sii'  # Replace with the name of your label column
if label_column in train_df.columns:
    # Check for invalid values (NaN, Infinity, -Infinity)
    invalid_values = train_df[~train_df[label_column].apply(np.isfinite)]  # Find invalid rows
    if not invalid_values.empty:
        print(f"Found invalid values in label column '{label_column}':")
        print(invalid_values)

    # Drop rows with invalid values
    train_df = train_df[train_df[label_column].apply(np.isfinite)]
# Готовим данные для AutoGluon: создаем датасеты
# В данном примере предположим, что колонка 'target' является целевой переменной
label_column = 'sii'  # Замените на имя вашей целевой переменной
train_data, test_data = train_test_split(train_df, test_size=0.2, random_state=42)

train_data = TabularDataset(train_data)
test_data = TabularDataset(test_data)


# Обучаем модели с использованием AutoGluon
print("Training models using AutoGluon...")
# model_save_path = 'saved_models/ag_model'
#
# predictor = TabularPredictor(label=label_column, eval_metric='accuracy',path=model_save_path).fit(train_data, time_limit=60, presets='best_quality', keep_only_best=True)  # time_limit=60 ограничивает время обучения 60 секундами
#
# # Сохраняем лучшую обученную модель
# predictor.save(model_save_path)
# print(f"Model saved to {model_save_path}")
#
# # Можно позже загрузить модель через:
# print(test_data)
predictor = TabularPredictor.load('saved_models/ag_model')





# Предикт для тестовых данных
if 'sii' in test_data.columns:  # Если в тестовых данных есть целевая переменная
    test_predictions = predictor.predict(test_data)
    print("Test Predictions:")
    print(test_predictions)
else:
    print("No target found in test dataset. Predictions skipped.")

# Вывод метрик (если известны истинные значения на тестовых данных)
if 'sii' in test_data.columns:
    test_labels = test_data[label_column]
    accuracy = predictor.evaluate_predictions(y_true=test_labels, y_pred=test_predictions)
    print(f"Test Accuracy: {accuracy}")
