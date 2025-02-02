from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
from preprocess import Preprocessor

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(random_state=42)
        self.preprocessor = Preprocessor()

    def train(self, data_path: Path):
        data = self.preprocessor.load_data(data_path)
        X_train, X_test, y_train, y_test = self.preprocessor.preprocess_data(data)

        # Extract feature names from the fitted pipeline
        feature_names = self.preprocessor.pipeline.get_feature_names_out()

        # Train model
        self.model.fit(X_train, y_train)

        # Save model, preprocessor, and feature names
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        joblib.dump(self.model, model_dir / "model.joblib")
        joblib.dump(self.preprocessor, model_dir / "preprocessor.joblib")
        joblib.dump(feature_names, model_dir / "feature_names.joblib")

        return self.model.score(X_test, y_test)

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train(Path("data/processed/cleaned_data.csv"))
