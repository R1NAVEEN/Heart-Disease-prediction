import streamlit as st
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

# Set page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon=":heart:",
    layout="wide"
)

# Custom background
st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://source.unsplash.com/collection/190727/1920x1080");
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.title("Heart Disease Prediction")

st.markdown(
    """
    This prediction model uses machine learning algorithms to provide accurate 
    predictions based on various health factors.
    """
)

# Load the data
@st.cache
def load_data():
    df = pd.read_csv("heart.csv")  # Update with your data path
    return df

df = load_data()

# Model training
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True),  # Enable probability estimates
    "Neural Network": MLPClassifier(),
    "Naive Bayes": GaussianNB(),
    "AdaBoost": AdaBoostClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

voting_clf = VotingClassifier([(name, model) for name, model in models.items()])
voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# User input
st.sidebar.title("Enter Candidate Information")
age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.radio("Sex", [0, 1])
cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
chol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar (> 120 mg/dl)", [0, 1])
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.sidebar.slider("ST Depression Induced by Exercise", 0.0, 6.2, 3.0)
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2])

# Make prediction
user_input = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
user_df = pd.DataFrame(user_input, columns=X.columns)

# Calculate probabilities for each base estimator
probs = np.zeros((len(models), 2))
for idx, (_, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    probs[idx] = model.predict_proba(user_df)

# Aggregate probabilities
prob_disease = probs[:, 1].mean() * 100  # Convert to percentage

# Display prediction probability and message
st.subheader("Prediction Probability")
st.write(f"The predicted probability of having heart disease is: {prob_disease:.2f}%")

# Determine message based on probability
if prob_disease < 20:
    message = "You have a low chance of having a heart attack. Keep up with a healthy lifestyle!"
elif prob_disease < 50:
    message = "You have a moderate chance of having a heart attack. Consider making some lifestyle changes and consulting with a healthcare professional."
else:
    message = "You have a high chance of having a heart attack. It's crucial to seek medical advice immediately and make significant lifestyle changes."

# Display message
st.subheader("Message")
st.write(message)

# Plot heart rate graph if there's a chance of disease
if prob_disease >= 50:
    # Generate sample heart rate data (dummy example)
    num_minutes = 60 * 24  # 24 hours
    heart_rate = np.random.normal(loc=80, scale=10, size=num_minutes)

    # Plot heart rate graph
    plt.figure(figsize=(10, 5))
    plt.plot(heart_rate, color='blue')
    plt.title('Heart Rate Variation')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Heart Rate')
    st.pyplot(plt)
