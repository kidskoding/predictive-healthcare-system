import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib

class Model:
    def __init__(self, dataset_path, target_column):
        self.dataset_path = dataset_path
        self.target_column = target_column
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        data = pd.read_csv(self.dataset_path)
        X = data.drop(self.target_column, axis=1)
        y = data[self.target_column]
        return X, y

    def preprocess_data(self, X):
        X = X.fillna(X.mean())
        return X

    def train_test_split_data(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        self.model = RandomForestClassifier(random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def test_model(self):
        if self.model is None:
            print("Model not trained yet!")
            return

        y_pred = self.model.predict(self.X_test)
        print("Accuracy Score:", accuracy_score(self.y_test, y_pred), end="\n\n")
        print("Classification Report:\n", classification_report(self.y_test, y_pred))

    def save_model(self):
        if self.model is None:
            print("Model is not trained yet!")
            return

        filename = "./model.joblib"
        joblib.dump(self.model, filename)
        print(f"Model saved to {filename}")

    def load_model(self):
        filename = "./model.joblib"
        self.model = joblib.load(filename)
        print(f"Model loaded from {filename}")
