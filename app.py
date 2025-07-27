import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the correctly trained balanced model
model = joblib.load("stream_predictor.pkl")

st.title("🎯 Career Stream Recommendation Quiz")
st.markdown("Answer the questions on a scale of 0 (Not at all) to 10 (Absolutely!)")

questions = [
    "How interested are you in working with technology and digital tools?",
    "How curious are you about business, startups, or marketing?",
    "How good are you at solving logic or math-related problems?",
    "How creative do you feel when generating new ideas or designs?",
    "How confident are you in taking initiative or leading a group?",
    "How well do you perform under pressure or tight deadlines?",
    "How much do you enjoy teaching or helping others understand new topics?",
    "How comfortable are you in social situations or working with people?",
    "How good are you at organizing tasks and creating plans?",
    "How effective are your communication skills (speaking or writing)?",
]

answers = []
for i, q in enumerate(questions):
    ans = st.slider(q, min_value=0, max_value=10, key=i)
    answers.append(ans)

user_input = np.array(answers).reshape(1, -1)

if st.button("Suggest Career Stream"):
    prediction = model.predict(user_input)[0]
    st.success(f"🌟 Based on your answers, a suitable career stream is: **{prediction}**")
