import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv("data/placement.csv")
df.columns = df.columns.str.strip()

df["placement_status"] = df["placement_status"].map({
    "Placed": 1,
    "Not Placed": 0
})

df = df.drop(["student_id", "salary_package_lpa"], axis=1)

df = pd.get_dummies(df, drop_first=True)

# Train model
X = df.drop("placement_status", axis=1)
y = df["placement_status"]

model = RandomForestClassifier()
model.fit(X, y)

# App title
st.title("Placement Prediction Dashboard")

st.write("Enter student details to predict placement chances")

# Inputs
cgpa = st.slider("CGPA", 0.0, 10.0, 7.0)
coding = st.slider("Coding Skill", 0, 100, 50)
aptitude = st.slider("Aptitude Score", 0, 100, 50)
communication = st.slider("Communication Skill", 0, 100, 50)
mock = st.slider("Mock Interview Score", 0, 100, 50)

# Prediction
if st.button("Predict"):

    input_data = pd.DataFrame([{
        "cgpa": cgpa,
        "coding_skill_score": coding,
        "aptitude_score": aptitude,
        "communication_skill_score": communication,
        "mock_interview_score": mock
    }])

    for col in X.columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data = input_data[X.columns]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Placed ✅ ({probability:.2%} confidence)")
    else:
        st.error(f"Not Placed ❌ ({1-probability:.2%} confidence)")