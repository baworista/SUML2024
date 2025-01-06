import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from autogluon.tabular import TabularDataset, TabularPredictor


class AutoGluonProcessor:
    def __init__(self, label_column, model_save_path="saved_models/ag_model", eval_metric="accuracy"):
        self.label_column = label_column
        self.model_save_path = model_save_path
        self.eval_metric = eval_metric

    def preprocess_data(self, data_path):
        """
        Load and preprocess the data.
        """
        df = pd.read_csv(data_path)
        if self.label_column not in df.columns:
            raise ValueError(f"Label column '{self.label_column}' is missing in the dataset.")

        # Check for invalid values
        invalid_values = df[~df[self.label_column].apply(np.isfinite)]
        if not invalid_values.empty:
            print(f"Found invalid values in label column '{self.label_column}':")
            print(invalid_values)
            df = df[df[self.label_column].apply(np.isfinite)]

        return df

    def train_model(self, train_df, time_limit=30):
        """
        Train an AutoGluon model on the provided dataset.
        """
        train_data, _ = train_test_split(train_df, test_size=0.2, random_state=42)
        train_data = TabularDataset(train_data)

        print("Training models using AutoGluon...")
        predictor = TabularPredictor(
            label=self.label_column,
            eval_metric=self.eval_metric,
            path=self.model_save_path
        ).fit(train_data, time_limit=time_limit, presets="best_quality", keep_only_best=True)

        print(f"✅ Model saved to {self.model_save_path}")

        # Print the features used by the model
        used_features = predictor._learner.feature_generator.features_in
        print("🛠️ **Features used by the model for training:**")
        for feature in used_features:
            print(f"- {feature}")

        return predictor

    def load_model(self):
        """
        Load a trained AutoGluon model from the path.
        """
        predictor = TabularPredictor.load(self.model_save_path)
        print("✅ Model loaded successfully!")

        # Print the features used by the model
        used_features = predictor._learner.feature_generator.features_in
        print("🛠️ **Features used by the loaded model:**")
        for feature in used_features:
            print(f"- {feature}")

        return predictor

    def predict(self, predictor, test_df):
        """
        Make predictions using the trained predictor.
        """
        if self.label_column in test_df.columns:
            true_labels = test_df[self.label_column]
            test_data = TabularDataset(test_df.drop(columns=[self.label_column]))
        else:
            true_labels = None
            test_data = TabularDataset(test_df)

        predictions = predictor.predict(test_data)

        if true_labels is not None:
            accuracy = predictor.evaluate_predictions(y_true=true_labels, y_pred=predictions)
            print(f"🎯 Test Accuracy: {accuracy}")
        else:
            print("⚠️ No target found in test dataset. Predictions made without true labels.")

        print("📊 Predictions:")
        print(predictions)
        return predictions
