import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from preprocess import Preprocessor
from train import ModelTrainer
from datetime import datetime

# Initialize session state variables
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []


@st.cache_resource
def load_model_and_preprocessor():
    try:
        model_dir = Path("models")
        model = joblib.load(model_dir / "model.joblib")
        preprocessor = joblib.load(model_dir / "preprocessor.joblib")
        return model, preprocessor
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first.")
        return None, None


def preprocess_single_prediction(preprocessor, data):
    """Preprocess a single prediction without train/test split"""
    # Apply the same preprocessing steps but without splitting
    data = preprocessor.feature_engineering(data)
    data = preprocessor.feature_selection(data)

    # Get numeric and categorical features
    numeric_features = data.select_dtypes(include=[np.number]).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    # Create column transformer
    from sklearn.compose import ColumnTransformer
    preprocessor_transform = ColumnTransformer(
        transformers=[
            ('num', preprocessor.scaler, numeric_features),
            ('cat', preprocessor.ohe, categorical_features)
        ]
    )

    return preprocessor_transform.fit_transform(data)


def make_prediction(model, preprocessor, input_data):
    if model is None or preprocessor is None:
        st.error("Model not loaded. Please train the model first.")
        return None

    try:
        # Add all required columns with dummy/default values
        input_data['session_id'] = 'test_session'
        input_data['user_id'] = 'test_user'
        input_data['campaign_id'] = 'test_campaign'
        input_data['webpage_id'] = 'test_webpage'
        input_data['is_click'] = 0  # Dummy value as this is what we're predicting

        # Change datetime to DateTime to match expected column name
        input_data = input_data.rename(columns={'datetime': 'DateTime'})

        st.write("Debug - Input Data Columns:", input_data.columns.tolist())

        # Use single prediction preprocessing instead of train/test split
        processed_features = preprocess_single_prediction(preprocessor, input_data)
        prediction = model.predict(processed_features)
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Debug - Full error:", str(e))
        return None


def main():
    st.title("DSiP ML Project Interface")

    model, preprocessor = load_model_and_preprocessor()

    if model is not None and preprocessor is not None:
        st.header("Make Predictions")

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                product = st.text_input("Product", "H")
                product_category_1 = st.selectbox("Product Category 1", range(1, 6))
                product_category_2 = st.selectbox("Product Category 2", range(1, 6))

            with col2:
                age_level = st.selectbox("Age Level", ['1', '2', '3', '4', '5'])
                gender = st.selectbox("Gender", ['Male', 'Female'])
                user_group = st.selectbox("User Group", ['1', '2', '3', '4', '5'])

            with col3:
                user_depth = st.slider("User Depth", 1, 3, 1)
                city_dev_index = st.slider("City Development Index", 0, 6, 1)
                var_1 = st.selectbox("Var 1", range(1, 6))

            date = st.date_input("Date")
            time = st.time_input("Time")

            submitted = st.form_submit_button("Predict")
            if submitted:
                datetime_str = datetime.combine(date, time).strftime('%Y-%m-%d %H:%M:%S')

                input_data = pd.DataFrame({
                    'datetime': [datetime_str],
                    'product': [product],
                    'gender': [gender],
                    'age_level': [age_level],
                    'user_group_id': [user_group],
                    'product_category_1': [product_category_1],
                    'product_category_2': [product_category_2],
                    'user_depth': [user_depth],
                    'city_development_index': [city_dev_index],
                    'var_1': [var_1]
                })

                st.write("Debug - Created DataFrame:", input_data)

                prediction = make_prediction(model, preprocessor, input_data)
                if prediction is not None:
                    st.success(f"Click Prediction: {'Yes' if prediction[0] == 1 else 'No'}")

                    # Add to prediction history with all relevant features
                    st.session_state.prediction_history.append({
                        'datetime': datetime_str,
                        'product': product,
                        'product_category_1': product_category_1,
                        'product_category_2': product_category_2,
                        'user_depth': user_depth,
                        'prediction': 'Yes' if prediction[0] == 1 else 'No'
                    })

        # Display prediction history
        if st.session_state.prediction_history:
            st.subheader("Prediction History")
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df)


if __name__ == "__main__":
    main()