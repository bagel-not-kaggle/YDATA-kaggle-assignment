import sys
import os
from pathlib import Path
import plotly.express as px
import numpy as np
import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier, Pool
from pathlib import Path

# Define paths
current_dir = Path(os.path.dirname(os.path.abspath(__file__)))  # Path to app/
parent_dir = current_dir.parent  # Path to YDATA-kaggle-assignment/

# Add the root directory to sys.path
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Debugging: Print paths
print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")
print(f"sys.path: {sys.path}")

# Now import preprocess
from preprocess import DataPreprocessor



class StreamlitApp:
    def __init__(self):
        st.set_page_config(
            page_title="CTR Prediction App",
            page_icon="🎯",
            layout="wide"
        )
        self.initialize_session_state()
        self.load_model()
        self.setup_sidebar()

    def initialize_session_state(self):
        """Initialize all session state variables"""
        for key, default_value in {
            'predictions': None,
            'probabilities': None,
            'threshold': 0.5,
            'debug_mode': False
        }.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def load_model(self):
        """Load the CatBoost model"""

        # Change from current_dir to parent_dir
        model_path = parent_dir / "models" / "best_model_catboost_newest.cbm"  # Updated path

        try:
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")

            self.model = CatBoostClassifier()
            self.model.load_model(model_path)
            self.feature_names = self.model.feature_names_
            st.sidebar.success("Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
            st.stop()

    def setup_sidebar(self):
        """Setup sidebar elements"""
        st.sidebar.write("Model Configuration")
        st.session_state.threshold = st.sidebar.slider(
            "Prediction Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.threshold,
            step=0.05,
            help="Threshold for converting probabilities to binary predictions"
        )
        st.session_state.debug_mode = st.sidebar.checkbox(
            "Debug Mode",
            value=st.session_state.debug_mode,
            help="Show additional debugging information"
        )

    def debug_log(self, message, data=None):
        """Log debug information if debug mode is enabled"""
        if st.session_state.debug_mode:
            st.write(message)
            if data is not None:
                st.write(data)

    def ensure_column_order(self, df):
        """Ensure columns are in the correct order based on model's feature names"""
        if not hasattr(self, 'feature_names') or not self.feature_names:
            raise ValueError("Model feature names not available")

        ordered_df = pd.DataFrame()
        for feature in self.feature_names:
            if feature in df.columns:
                ordered_df[feature] = df[feature]
            else:
                self.debug_log(f"Missing feature: {feature}")
                ordered_df[feature] = "missing"
        return ordered_df

    def prepare_features(self, df, cat_features):
        """Prepare features for CatBoost"""
        df = df.copy()
        df = self.ensure_column_order(df)

        cat_indices = []
        for i, feature in enumerate(self.feature_names):
            if feature in cat_features:
                cat_indices.append(i)
                df[feature] = df[feature].astype(str)
            else:
                if feature in df.columns and df[feature].dtype == 'object':
                    df[feature] = df[feature].astype(str)
                    mask = df[feature] == 'missing'
                    df.loc[mask, feature] = np.nan
                    df[feature] = pd.to_numeric(df[feature], errors='coerce')
                    df[feature] = df[feature].fillna(0)

        self.debug_log("Feature names from model:", self.feature_names)
        self.debug_log("Actual columns in data:", df.columns.tolist())
        self.debug_log("Categorical feature indices:", cat_indices)
        self.debug_log("Data types after preparation:", df.dtypes)

        return df, cat_indices

    def preprocess_test_data(self, df):
        """Preprocess test data using the preprocessor with enhanced logging"""
        try:
            # Create a placeholder for logging
            log_container = st.empty()

            def logging_callback(message):
                """Callback to display preprocessing logs in Streamlit"""
                if st.session_state.debug_mode:
                    with log_container.container():
                        st.text(message)

            preprocessor = DataPreprocessor(
                output_path=current_dir / "data" / "processed",  # Updated path
                remove_outliers=False,
                fillna=True,
                save_as_pickle=False,
                callback=logging_callback
            )

            processed_df = preprocessor.preprocess_test(df_test=df)

            if st.session_state.debug_mode:
                st.write("Preprocessing Statistics:")
                stats = {
                    "Original Shape": df.shape,
                    "Processed Shape": processed_df.shape,
                    "Missing Values": processed_df.isnull().sum().sum(),
                    "Categorical Columns": len(processed_df.select_dtypes(include=['object', 'category']).columns),
                    "Numeric Columns": len(processed_df.select_dtypes(include=['int64', 'float64']).columns)
                }
                st.json(stats)

            return processed_df

        except Exception as e:
            st.error(f"Error in preprocessing: {str(e)}")
            raise

    def display_predictions(self):
        """Display predictions and statistics"""
        predictions_df = pd.DataFrame(st.session_state.predictions, columns=['is_click'])
        probabilities_df = pd.DataFrame(st.session_state.probabilities, columns=['click_probability'])

        # Binary Predictions Section
        st.subheader("Binary Predictions (First 5 rows)")
        st.write(predictions_df.head())
        st.download_button(
            label="Download Binary Predictions",
            data=predictions_df.to_csv(index=False, header=False),
            file_name="predictions_binary.csv",
            mime="text/csv"
        )

        # Probability Predictions Section
        st.subheader("Probability Predictions (First 5 rows)")
        st.write(probabilities_df.head())
        st.download_button(
            label="Download Probability Predictions",
            data=probabilities_df.to_csv(index=False, header=False),
            file_name="predictions_probability.csv",
            mime="text/csv"
        )

        # Statistics Section
        st.subheader("Prediction Statistics")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Binary Predictions:")
            st.write(f"Number of predictions: {len(st.session_state.predictions)}")
            st.write(f"Mean prediction (threshold={st.session_state.threshold:.2f}): "
                     f"{st.session_state.predictions.mean():.4f}")

        with col2:
            st.write("Probability Predictions:")
            st.write(f"Mean probability: {st.session_state.probabilities.mean():.4f}")
            st.write(f"Probability range: {st.session_state.probabilities.min():.4f} "
                     f"to {st.session_state.probabilities.max():.4f}")

        # Distribution Plots
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Binary Prediction Distribution")
            hist_data = pd.DataFrame(st.session_state.predictions, columns=['Click'])
            click_counts = hist_data['Click'].value_counts().sort_index()
            st.bar_chart(click_counts)

        with col2:
            st.subheader("Probability Distribution")
            hist_data = pd.DataFrame(st.session_state.probabilities, columns=['Probability'])
            bins = np.linspace(0, 1, 11)
            hist_values, _ = np.histogram(hist_data['Probability'], bins=bins)
            hist_df = pd.DataFrame({
                'Probability Range': [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)],
                'Count': hist_values
            })
            hist_df = hist_df.set_index('Probability Range')
            st.bar_chart(hist_df)

    def generate_predictions(self, df):
        """Generate predictions with enhanced logging and analysis"""
        try:
            with st.expander("Preprocessing Details", expanded=st.session_state.debug_mode):
                processed_df = self.preprocess_test_data(df)
            preprocess = DataPreprocessor()
            cat_features = preprocess.determine_categorical_features(processed_df)
            processed_df, cat_indices = self.prepare_features(processed_df, cat_features)
            
            if st.session_state.debug_mode:
                st.write("Feature Preparation Summary:")
                st.write({
                    "Total Features": len(self.feature_names),
                    "Categorical Features": len(cat_indices),
                    "Numeric Features": len(self.feature_names) - len(cat_indices)
                })

            test_pool = Pool(data=processed_df, cat_features=cat_indices)

            with st.spinner("Generating predictions..."):
                probabilities = self.model.predict_proba(test_pool)[:, 1]
                predictions = (probabilities >= st.session_state.threshold).astype(int)

                st.session_state.predictions = predictions
                st.session_state.probabilities = probabilities

                # Display feature importance analysis
                self.display_feature_importance(processed_df)

        except Exception as e:
            st.error(f"Error generating predictions: {str(e)}")
            if st.session_state.debug_mode:
                st.error("Debug info:")
                st.write("Model type:", type(self.model))
                if 'processed_df' in locals():
                    st.write("Processed data shape:", processed_df.shape)
                    st.write("Processed data columns:", processed_df.columns.tolist())
            raise

    def display_feature_importance(self, processed_df):
        """Display feature importance analysis"""
        st.subheader("Feature Importance Analysis")

        try:
            # Get feature importance from the model
            importance = self.model.get_feature_importance()
            feature_names = processed_df.columns

            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

            # Display top features
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write("Top 20 Most Important Features")
                # Create bar chart
                import plotly.express as px
                fig = px.bar(
                    importance_df.head(20),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Feature Importance'
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.write("Feature Importance Statistics")
                # Categorize features
                importance_df['feature_type'] = importance_df['feature'].apply(
                    lambda x: 'CTR' if '_ctrS' in x
                    else 'Target Encoded' if '_te' in x
                    else 'Time-based' if any(t in x.lower() for t in ['day', 'hour', 'minute', 'weekday'])
                    else 'Duration-based' if 'duration' in x.lower()
                    else 'Other'
                )

                # Show importance by feature type
                type_importance = importance_df.groupby('feature_type')['importance'].sum().sort_values(ascending=False)
                st.write("Importance by Feature Type:")
                st.dataframe(type_importance)

                # Show basic statistics
                st.write("Summary Statistics:")
                stats = pd.DataFrame({
                    'Metric': ['Number of features', 'Mean importance', 'Median importance'],
                    'Value': [
                        len(importance_df),
                        f"{importance_df['importance'].mean():.4f}",
                        f"{importance_df['importance'].median():.4f}"
                    ]
                })
                st.dataframe(stats)

        except Exception as e:
            st.error(f"Error analyzing feature importance: {str(e)}")
    def home_page(self):
        """Main page with prediction functionality"""
        st.title("🎯 Click Prediction Model")
        st.write("""
        Upload your test data file to get click predictions.
        The file should be a CSV containing the required features for prediction.
        """)

        uploaded_file = st.file_uploader("Upload test data (CSV)", type=['csv'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.subheader("Data Preview")
                st.write(df.head())
                st.write("Data shape:", df.shape)
                self.debug_log("Original columns:", df.columns.tolist())

                if st.button("Generate Predictions"):
                    with st.spinner("Preprocessing data and generating predictions..."):
                        self.generate_predictions(df)

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                raise

        if st.session_state.predictions is not None:
            self.display_predictions()

    def run(self):
        """Run the Streamlit app"""
        self.home_page()


if __name__ == "__main__":
    app = StreamlitApp()
    app.run()