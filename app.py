import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the model, label encoder, and scaler
model = joblib.load("career_model.joblib")
le_interest = joblib.load("interest_encoder.joblib")
scaler = joblib.load("scaler.joblib")

# Streamlit UI
st.set_page_config(page_title="Career Suggestion for Rural Students", layout="centered")
st.title("🎯 Career Suggestion Model")
st.markdown("Get personalized career suggestions based on your **interests** and **aptitude** scores!")

# Interest options
interest_options = le_interest.classes_.tolist()

# --- INPUT FORM ---
with st.form("career_form"):
    interest = st.selectbox("🧠 What subject or field interests you the most?", interest_options)

    logical = st.slider("🧩 Logical Thinking (problem-solving skills)", 1, 10, 5)
    communication = st.slider("🗣️ Communication Skills", 1, 10, 5)
    leadership = st.slider("👑 Leadership Skills", 1, 10, 5)
    creativity = st.slider("🎨 Creativity Level", 1, 10, 5)

    submit = st.form_submit_button("Find My Career 🚀")

# --- PREDICTION ---
if submit:
    # Encode interest
    interest_encoded = le_interest.transform([interest])[0]

    # Prepare input data
    input_df = pd.DataFrame([{
        "Interest": interest_encoded,
        "Logical_Thinking": logical,
        "Communication": communication,
        "Leadership": leadership,
        "Creativity": creativity
    }])

    # Scale the aptitude scores
    input_df[["Logical_Thinking", "Communication", "Leadership", "Creativity"]] = scaler.transform(
        input_df[["Logical_Thinking", "Communication", "Leadership", "Creativity"]]
    )

    # Predict probabilities for top-3 career options
    probabilities = model.predict_proba(input_df)[0]
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_careers = [model.classes_[i] for i in top_3_indices]
    top_3_scores = [probabilities[i] * 100 for i in top_3_indices]

    # Show result
    st.success("🎓 Based on your profile, your top career suggestions are:")
    for i in range(3):
        st.write(f"{i+1}. **{top_3_careers[i]}** — {top_3_scores[i]:.2f}% match")
