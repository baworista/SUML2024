import streamlit as st
import pandas as pd
from autogluon.tabular import TabularPredictor
import io  # Для обработки CSV-строки как файла
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Paths to the saved model
MODEL_PATH = "/saved_models/ag_model"


# Load the model
@st.cache_resource
def load_model():
    st.write("📥 Loading the model...")
    predictor = TabularPredictor.load(MODEL_PATH, require_version_match=False)
    st.write("✅ Model loaded successfully!")
    return predictor


# Align input data with model's expected features
def align_features(input_data, expected_features):
    """
    Align input data with the expected features from the model.
    Missing features are filled with default values (0 or NaN).
    Extra features are removed.
    """
    # Удаляем столбцы с Unnamed
    input_data = input_data.loc[:, ~input_data.columns.str.contains('^Unnamed')]

    # Заполняем недостающие признаки
    missing_features = set(expected_features) - set(input_data.columns)
    for feature in missing_features:
        input_data[feature] = 0  # Заполнение значением по умолчанию (0)

    # Удаляем лишние признаки
    extra_features = set(input_data.columns) - set(expected_features)
    if extra_features:
        st.warning(f"⚠️ Extra features ignored: {extra_features}")
        input_data = input_data.drop(columns=list(extra_features))

    # Сортируем признаки в правильном порядке
    input_data = input_data[expected_features]

    return input_data


# Run prediction in a separate thread
def predict_fn(predictor, input_data_transformed):
    """Runs the prediction in a separate thread."""
    return predictor.predict(input_data_transformed)


# Streamlit UI
def main():
    st.title("🔮 AutoGluon Model Prediction App")
    st.write("Enter feature values as a **CSV string**.")

    predictor = load_model()
    used_features = predictor._learner.feature_generator.features_in

    # Display expected features
    with st.expander("🛠️ Features Required by the Model"):
        st.write(used_features)

    # User input: CSV string
    user_input = st.text_area(
        "📝 Enter feature values as CSV (header is optional, values separated by commas):",
        placeholder="Basic_Demos-Enroll_Season,Basic_Demos-Age,...\nFall,5,..."
    )

    if st.button("🔍 Make Prediction"):
        if not user_input:
            st.warning("⚠️ Please enter CSV feature values.")
            return

        with st.spinner("⏳ Processing input and making prediction..."):
            try:
                # Parse CSV input
                csv_data = io.StringIO(user_input)
                try:
                    input_data = pd.read_csv(csv_data)
                except pd.errors.ParserError:
                    st.warning("⚠️ No headers detected, adding expected headers automatically.")
                    csv_data.seek(0)
                    input_data = pd.read_csv(csv_data, header=None)
                    input_data.columns = used_features[:len(input_data.columns)]

                # Align features
                input_data = align_features(input_data, used_features)
                st.write("✅ Input Data Aligned with Model Features:")
                st.dataframe(input_data)

                # Preprocess input data
                st.write("⚙️ Preprocessing input data...")
                input_data_transformed = predictor.transform_features(input_data)
                st.write("✅ Data Transformed Successfully:")
                st.dataframe(input_data_transformed)

                # Prediction with timeout using thread
                st.write("🔮 Making prediction...")
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(predict_fn, predictor, input_data_transformed)
                    try:
                        prediction = future.result(timeout=10)  # Set timeout (in seconds)
                        st.success(f"✅ Prediction: {prediction.values[0]}")
                    except TimeoutError:
                        st.error("❌ Prediction took too long (timeout occurred). Try again later.")

            except pd.errors.EmptyDataError:
                st.error("❌ Invalid CSV format. Please check your input.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
