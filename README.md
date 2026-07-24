# 🏦 Smart Lender — AI-Powered Loan Approval Prediction System

🔗 **[Live Demo](https://smart-lender-v7wz.onrender.com)** *(may take 30-60 sec to load if idle)*

An end-to-end machine learning web application that predicts whether a loan application is likely to be **approved or rejected**, based on applicant details such as income, credit history, education, and property area. Built with **Python, scikit-learn, XGBoost, and Flask**.

---

## 📌 Problem Statement

Manual loan approval processes are slow and inconsistent. This project uses historical loan application data to train a machine learning model that predicts loan approval outcomes in real time, giving both applicants and lenders a fast, data-driven decision along with a confidence score.

---

## 🚀 Live Demo / Screenshots

| Landing Page | Prediction Form | Result |
|---|---|---| 
| <img width="300" alt="landing" src="https://github.com/user-attachments/assets/4897ee51-1e00-429a-a7a4-f14a0bc157d9" /> | <img width="300" alt="form" src="https://github.com/user-attachments/assets/13ffe67b-66eb-414c-afb5-01f583931edb" /> |<img width="300" alt="result" src="https://github.com/user-attachments/assets/2713f717-fd60-45d1-a65d-33d97e3208a0" /> |
---

## 🧠 Machine Learning Pipeline

**1. Exploratory Data Analysis (EDA)**
Analyzed loan status distribution across gender, marital status, education, income, and property area to understand key patterns in the dataset.

<img width="540" height="393" alt="loan_status_dist" src="https://github.com/user-attachments/assets/5c73cc7f-5bbd-4aff-958e-06c00f9eab95" />


**2. Data Preprocessing**
- Handled missing values (mode imputation for categorical, mean for numerical)
- Removed outliers from `ApplicantIncome` and `LoanAmount` using the IQR method
- Label-encoded categorical features
- Applied **StandardScaler** for feature scaling

**3. Handling Class Imbalance**
Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance the training data, since loan approvals significantly outnumbered rejections in the raw dataset.

**4. Model Building & Comparison**
Trained and compared four classification models on the SMOTE-balanced, scaled training data:

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Decision Tree | 1.000 | 0.748 |
| Random Forest | 1.000 | 0.832 |
| K-Nearest Neighbors | 0.859 | 0.710 |
| **XGBoost** | 1.000 | 0.822 |

While Random Forest showed marginally higher raw test accuracy, **XGBoost was selected as the final model** for deployment due to its strong, consistent performance, better handling of feature interactions, gradient-boosted robustness, and its widespread reliability in real-world classification tasks. The final model's precision, recall, and F1-score (see confusion matrix below) were used as the deciding evaluation criteria, not accuracy alone — since accuracy alone can be misleading with class imbalance even after SMOTE.


**5. Model Evaluation**
Evaluated the final XGBoost model using a confusion matrix and full classification report (precision, recall, F1-score) rather than accuracy alone, to properly account for class imbalance.

<img width="435" height="393" alt="confusion_matrix" src="https://github.com/user-attachments/assets/af18d56e-96c3-4f95-8854-dacea2818537" />


**6. Feature Importance**
Analyzed which features most influenced predictions. **Credit History** emerged as by far the most dominant factor, followed by **Property Area** — consistent with real-world lending practices where repayment history is the primary risk signal.

<img width="600" alt="Feature Importance" src="https://github.com/user-attachments/assets/ec861b1e-9497-420a-b463-0ed8de05b353" />


---

## 🛠️ Tech Stack

- **Language:** Python
- **ML/Data:** pandas, numpy, scikit-learn, XGBoost, imbalanced-learn (SMOTE)
- **Visualization:** matplotlib, seaborn
- **Web Framework:** Flask
- **Frontend:** HTML, CSS (Jinja2 templates)

---

## 📂 Project Structure

```
smart-lender-loan-prediction/
├── dataset/
│   └── loan_data.csv
├── models/
│   ├── encoders.pkl
│   ├── scaler.pkl
│   └── rdf.pkl
├── templates/
│   ├── home.html
│   ├── predict.html
│   └── result.html
├── smart_lender.ipynb
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/poojithapeddiboina/smart-lender-loan-prediction.git
   cd smart-lender-loan-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask app**
   ```bash
   python app.py
   ```

4. **Open in browser**
   ```
   http://127.0.0.1:5000
   ```

---

## 🔑 Key Insight

The model's decisions are driven overwhelmingly by **Credit History** — applicants with a good repayment history are approved with high confidence regardless of minor variations in other fields, while a poor credit history strongly pushes the model toward rejection. This mirrors how real financial institutions assess risk.

<img width="695" height="393" alt="property_area" src="https://github.com/user-attachments/assets/76ea6262-6d29-4cad-848f-9d239d753e20" />


---

## 📈 Future Improvements

- Add SHAP-based explainability per prediction (why *this specific* application was approved/rejected)
- Expand dataset for better generalization
- Add authentication for a lender-facing dashboard
- Add input validation on the frontend form

---

## 👤 Author

**Venkata Poojitha Peddi Boina**
[GitHub](https://github.com/poojithapeddiboina)
