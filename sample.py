# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load("career_model_detailed.pkl")

st.title("🎯 Career Stream Recommendation Quiz")
st.markdown("Rate each question from 0 (Not at all) to 5 (Very much) based on your interest.")

# Define the questions (same as in the dataset)
questions = [
    "Do you enjoy solving complex problems using logic or mathematics?",  # Q1
    "Do you enjoy working on tasks that require originality and artistic flair?",  # Q2
    "Are you comfortable working with large datasets, statistics, or analysis?",  # Q3
    "Do you prefer hands-on technical tasks over theory?",  # Q4
    "Are you curious about how living things work or how life evolves?",  # Q5
    "Do you like working with technology and constantly learning about new tools?",  # Q6
    "Do you enjoy building or designing machines, systems, or structures?",  # Q7
    "Are you passionate about drawing, designing, or expressing visually?",  # Q8
    "Are you fascinated by systems that involve rules, justice, or societal order?",  # Q9
    "Do you care deeply about environmental or health-related issues?",  # Q10
    "Do you enjoy analyzing markets, finances, or investment trends?",  # Q11
    "Do you enjoy conducting experiments or research in controlled settings?",  # Q12
    "Are you interested in designing digital products or visuals (e.g., games, websites)?",  # Q13
    "Do you enjoy leading teams or taking initiative in group projects?",  # Q14
    "Are you interested in starting your own business someday?",  # Q15
    "Do you feel confident speaking in front of groups or communicating ideas?",  # Q16
    "Do you find joy in helping others understand and learn new things?",  # Q17
    "Are you a good listener and able to empathize with people’s problems?",  # Q18
    "Do you value collaboration over working alone?",  # Q19
    "Would you enjoy working in high-pressure situations requiring precision and responsibility (e.g., flying, surgeries)?"  # Q20
]
# Collect answers via sliders
answers = []
for i, q in enumerate(questions):
    val = st.slider(q, 0, 5, key=f"q{i}")
    answers.append(val)

# Convert to model input
user_input = np.array(answers).reshape(1, -1)

# Predict and show output
if st.button("🔍 Suggest Career Stream"):
    prediction = model.predict(user_input)[0]
    st.success(f"🌟 Based on your inputs, a suitable career stream is: **{prediction}**")
