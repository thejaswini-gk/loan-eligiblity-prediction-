import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv('loan_data.csv')

# Handle missing values
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['ApplicantIncome'].fillna(df['ApplicantIncome'].mean(), inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

# Encoding categorical variables (Gender, Married) as 0 and 1
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})

# Ensure Loan_Status is binary (1=Approved, 0=Rejected)
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Select relevant features and target variable
X = df[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount']]
y = df['Loan_Status']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the trained model to a file
with open('loan_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
print("Model saved as loan_model.pkl")
