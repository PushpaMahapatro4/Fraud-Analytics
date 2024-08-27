import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('/mnt/data/df.csv')
    return data

# App title
st.title("Fraud Analytics on Transaction Dataset")

# Load data
data = load_data()

# Display the first few rows of the dataset
st.subheader("Dataset Overview")
st.write(data.head())

# Display dataset statistics
st.subheader("Dataset Statistics")
st.write(data.describe())

# Display class distribution
st.subheader("Class Distribution")
st.write(data['isFraud'].value_counts())

# Plot class distribution
st.subheader("Fraud vs. Non-Fraud Transactions")
fig, ax = plt.subplots()
sns.countplot(x='isFraud', data=data, ax=ax)
st.pyplot(fig)

# Feature correlation heatmap
st.subheader("Feature Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Encode categorical features
le = LabelEncoder()
data['type'] = le.fit_transform(data['type'])

# Select relevant features and target
features = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
X = data[features]
y = data['isFraud']

# Train a simple Random Forest model
st.subheader("Train a Model")
model = None
if st.button('Train Model'):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the Random Forest model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Display model performance
    st.subheader("Model Performance")
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Display the confusion matrix
    st.write("Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Option to download the model
    st.subheader("Download Model")
    if st.button('Download Trained Model'):
        with open('fraud_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        st.write("Model downloaded as `fraud_model.pkl`.")
