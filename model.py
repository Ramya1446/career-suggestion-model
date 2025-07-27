from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd

df = pd.read_csv("career_dataset_balanced_20Q.csv")
X = df.drop(columns=["Career"])
y = df["Career"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=300, max_depth=15, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

# Save model
import joblib
joblib.dump(model, "career_model_detailed.pkl")

# Test prediction accuracy
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
