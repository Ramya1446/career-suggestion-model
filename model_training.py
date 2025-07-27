import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model():
    df = pd.read_csv('balanced_career_dataset_500.csv')

    X = df.drop(columns=['Stream', 'SuitableCareer'])
    y = df['Stream']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, 'stream_predictor.pkl')

    print("Model trained. Accuracy on test set:", model.score(X_test, y_test))

if __name__ == "__main__":
    train_and_save_model()
