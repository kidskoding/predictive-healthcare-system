import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(dataset_path, target_column):
    data = pd.read_csv(dataset_path)
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return X, y

def preprocess_data(X):
    X = X.fillna(X.mean())
    return X

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy Score:", accuracy_score(y_test, y_pred), end="\n\n")
    print("Classification Report:\n", classification_report(y_test, y_pred))

def save_model(model):
    filename = "./model.joblib"
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def visualize_class_distribution(y):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y)
    plt.title('Class Distribution')
    plt.show()
