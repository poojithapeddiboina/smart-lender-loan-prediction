from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load model, encoders, scaler
with open('models/rdf.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET'])
def predict_form():
    return render_template('predict.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Read form values
    gender = request.form['Gender']
    married = request.form['Married']
    dependents = request.form['Dependents']
    education = request.form['Education']
    self_employed = request.form['Self_Employed']
    applicant_income = float(request.form['ApplicantIncome'])
    coapplicant_income = float(request.form['CoapplicantIncome'])
    loan_amount = float(request.form['LoanAmount'])
    loan_amount_term = float(request.form['Loan_Amount_Term'])
    credit_history = float(request.form['Credit_History'])
    property_area = request.form['Property_Area']

    # Encode categorical fields using saved encoders
    gender_enc = encoders['Gender'].transform([gender])[0]
    married_enc = encoders['Married'].transform([married])[0]
    dependents_enc = encoders['Dependents'].transform([dependents])[0]
    education_enc = encoders['Education'].transform([education])[0]
    self_employed_enc = encoders['Self_Employed'].transform([self_employed])[0]
    property_area_enc = encoders['Property_Area'].transform([property_area])[0]

   # Build feature DataFrame with same column names/order as training
    feature_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
                        'Loan_Amount_Term', 'Credit_History', 'Property_Area']

    features = pd.DataFrame([[
        gender_enc, married_enc, dependents_enc, education_enc,
        self_employed_enc, applicant_income, coapplicant_income,
        loan_amount, loan_amount_term, credit_history, property_area_enc
    ]], columns=feature_columns)

    # Scale features (model was trained on scaled + SMOTE-balanced data)
    features_scaled = scaler.transform(features)

    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    confidence = round(max(probability) * 100, 2)

    result = "Approved" if prediction == 1 else "Rejected"

    return render_template('result.html', result=result, confidence=confidence)

if __name__ == '__main__':
    app.run(debug=True)