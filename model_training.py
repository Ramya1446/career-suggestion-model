import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load the dataset
df = pd.read_csv("career_dataset.csv")

# 2. Define career category mapping
career_mapping = {
    "AI Research Scientist": "Tech",
    "Software Developer": "Tech",
    "Cybersecurity Analyst": "Tech",
    "Full Stack Developer": "Tech",
    "Cloud Solutions Architect": "Tech",
    "Data Engineer": "Tech",
    "Data Scientist": "Tech",
    "DevOps Engineer": "Tech",

    "Clinical Psychologist": "Mental Health",
    "Psychologist": "Mental Health",
    "Therapist": "Mental Health",
    "Counselor": "Mental Health",
    "Child Psychologist": "Mental Health",
    "Forensic Psychologist": "Mental Health",
    "Behavioral Analyst": "Mental Health",

    "Lawyer": "Law",
    "Judge": "Law",
    "Public Prosecutor": "Law",
    "Legal Advisor": "Law",
    "Legal Researcher": "Law",
    "Corporate Attorney": "Law",
    "Paralegal": "Law",

    "Novelist": "Arts",
    "Poet": "Arts",
    "Fine Artist": "Arts",
    "Illustrator": "Arts",
    "Animator": "Arts",
    "3D Modeler": "Arts",
    "Art Director": "Arts",
    "Copywriter": "Arts",

    "Marketing Manager": "Business",
    "Sales Executive": "Business",
    "Product Manager": "Business",
    "HR Manager": "Business",
    "Business Analyst": "Business",
    "Content Writer": "Business",
    "Entrepreneur": "Business",

    "Biotechnologist": "Science",
    "Chemist": "Science",
    "Physicist": "Science",
    "Astronomer": "Science",
    "Environmental Scientist": "Science",
    "Lab Technician": "Science",

    "Journalist": "Media",
    "Editor": "Media",
    "Linguist": "Media"
}

# 3. Apply mapping
df["Career_Category"] = df["Suitable_Career"].map(career_mapping)

# Drop unmapped rows
df = df.dropna(subset=["Career_Category"])

# 4. Encode 'Interest' column
le_interest = LabelEncoder()
df['Interest'] = le_interest.fit_transform(df['Interest'])

# 5. Features and labels
X = df.drop(["Suitable_Career", "Career_Category"], axis=1)
y = df["Career_Category"]

# 6. Scale numerical features
scaler = MinMaxScaler()
X[["Logical_Thinking", "Communication", "Leadership", "Creativity"]] = scaler.fit_transform(
    X[["Logical_Thinking", "Communication", "Leadership", "Creativity"]]
)

# 7. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 9. Evaluation
y_pred = model.predict(X_test)
print("✅ Accuracy on test set:", accuracy_score(y_test, y_pred))
print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred))

# 10. Save model and tools
joblib.dump(model, "career_model.joblib")
joblib.dump(le_interest, "interest_encoder.joblib")
joblib.dump(scaler, "scaler.joblib")

print("✅ Model trained and saved successfully!")
