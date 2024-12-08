import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify, render_template
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
import os

app = Flask(__name__)

# Load the model
model = pickle.load(open('loan_model.pkl', 'rb'))

# Load and preprocess the dataset
df = pd.read_csv('loan_data.csv')

# Handle missing values
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
df['ApplicantIncome'] = df['ApplicantIncome'].fillna(df['ApplicantIncome'].mean())
df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].mean())

# Encode categorical features
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})

# Ensure Loan_Status is binary (1=Approved, 0=Rejected)
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Select features and target variable
X = df[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount']]
y = df['Loan_Status']

# Train LIME explainer
explainer = LimeTabularExplainer(X.values, feature_names=X.columns, class_names=['Rejected', 'Approved'], discretize_continuous=True)

# Function to generate and save graphs dynamically
def generate_graphs(prediction):
    # Create a directory to save images
    if not os.path.exists('static/images'):
        os.makedirs('static/images')

    # Bar plot of the prediction result (Approved or Rejected)
    plt.figure(figsize=(6, 4))
    sns.countplot(x=[prediction], palette=['green' if prediction == 'Approved' else 'red'])
    plt.title('Loan Status Prediction')
    plt.xticks([0], [prediction])
    plt.savefig('static/images/prediction_result_bar.png')
    plt.close()

    # Pie chart for Gender Distribution
    plt.figure(figsize=(6, 6))
    df['Gender'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['skyblue', 'orange'])
    plt.title('Gender Distribution')
    plt.savefig('static/images/gender_pie.png')
    plt.close()

    # Pie chart for Married Distribution
    plt.figure(figsize=(6, 6))
    df['Married'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
    plt.title('Marital Status Distribution')
    plt.savefig('static/images/married_pie.png')
    plt.close()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    # Make prediction
    prediction = model.predict(final_features)
    output = 'Approved' if prediction == 1 else 'Rejected'
    
    # Generate dynamic graphs based on prediction result
    generate_graphs(output)
    
    # LIME explanation
    exp = explainer.explain_instance(final_features[0], model.predict_proba, num_features=4)
    exp_html = exp.as_html()

    return render_template('result.html', prediction_text=f'Loan Status: {output}', explanation=exp_html)

if __name__ == "__main__":
    app.run(debug=True)
