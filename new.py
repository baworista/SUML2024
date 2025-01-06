from autogluon.tabular import TabularPredictor
import pandas as pd
import signal  # For timeout handling


# Timeout handler for long-running predictions
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


# Paths to the saved model and train CSV file
model_path = "/saved_models/ag_model"
train_file_path = '/data/train.csv'  # Use training data file

# Step 1. Load the model
print("📥 Loading the model...")
predictor = TabularPredictor.load(model_path)
print("✅ Model loaded successfully!")

# Step 2. Display the features used by the model
print("🛠️ Features used by the model:")
used_features = predictor._learner.feature_generator.features_in
print(used_features)

# Step 3. Load training data without the target column
print("📂 Loading training data...")
train_data = pd.read_csv(train_file_path)

target_column = predictor._learner.label
if target_column in train_data.columns:
    print(f"🗑️ Dropping target column: {target_column}")
    input_data = train_data.drop(columns=[target_column])
else:
    input_data = train_data.copy()

# Select a few rows for prediction
raw_input_data = input_data.iloc[:2].copy()

# Step 4. Preprocess and ensure no missing columns
print("🔍 Checking for missing columns...")
required_columns = predictor.feature_metadata.get_features()
current_columns = set(raw_input_data.columns)
missing_columns = set(required_columns) - current_columns

if missing_columns:
    print(f"⚠️ Adding missing columns: {missing_columns}")
    for col in missing_columns:
        raw_input_data[col] = 0
else:
    print("✅ No missing columns found.")

# Preprocess input data
print("⚙️ Preprocessing raw input data...")
input_data_transformed = predictor.transform_features(raw_input_data)
print("✅ Data transformed successfully!")
print("🔍 Transformed Data for Model:")
print(input_data_transformed)

# Step 5. Display the actual features being used in prediction
print("📊 Actual features used in prediction:")
print(input_data_transformed.columns.tolist())

# Step 6. Predict with timeout
print("🔮 Making predictions with timeout...")
try:
    signal.signal(signal.SIGALRM, timeout_handler)  # Set the timeout handler
    signal.alarm(10)  # Set timeout to 10 seconds for predictions

    for idx, row in input_data_transformed.iterrows():
        print(f"📊 Making prediction for Row {idx + 1}...")

        # Predict a single row
        single_row = row.to_frame().T
        pred = predictor.predict(single_row)

        # Cancel alarm if successful
        signal.alarm(0)
        print(f"✅ Prediction for Row {idx + 1}: {pred.values[0]}")
except TimeoutException:
    print("❌ Prediction took too long (timeout occurred).")
except Exception as e:
    print(f"❌ Error during prediction: {e}")
finally:
    signal.alarm(0)  # Disable alarm in any case

print("✅ Script completed.")
