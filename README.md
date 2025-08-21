
# Walmart Customer Churn Prediction

A Streamlit-based web application that predicts customer churn risk using Walmart’s customer purchase behavior dataset. The project leverages machine learning models (Random Forest & Logistic Regression) and provides interactive dashboards for churn analysis.

# Features

📂 Load dataset from Kaggle (via kagglehub) or upload your own CSV file

🧹 Automatic data preprocessing:
- Date handling
- Recency calculation
- Churn labels (Churn_Repeat, Churn_Recency)

🤖 Train multiple ML models (Random Forest, Logistic Regression) and compare accuracies

📊 Interactive dashboards with:
- Confusion matrices
- Risk distribution histograms
- Customer risk ranking tables

📥 Export top 100 at-risk customers as CSV files

📊 Types of Churn Analysis
- Repeat Customer Churn – Customers marked as   churned if not repeat buyers
- Recency Churn – Customers inactive for more than 90 days

Combined Risk Analysis – Weighted churn risk from both approaches

# Project Structure

├── WalmartCustomerChurnPrediction.py   
├── README.md                           
└── requirements.txt                 

# Installation & Setup

    1. Clone the Repository
    git clone https://github.com/srishtiiii-rana/Walmart-Customer-Churn-Prediction.git
    cd Walmart-Customer-Churn-Prediction

    2. Install Dependencies
    pip install -r requirements.txt
    
    Dependencies include:
    - streamlit
    - pandas, numpy
    - scikit-learn
    - plotly
    - kagglehub

    3. Run the App
    streamlit run WalmartCustomerChurnPrediction.py

# Example Outputs

- Model Accuracy Comparison
View side-by-side accuracy scores of Random Forest and Logistic Regression.

- Confusion Matrix
Interactive confusion matrix plots for performance evaluation.

- Top 100 At-Risk Customers
Downloadable CSV reports of customers most likely to churn.

# Dataset

Source: Walmart Customer Purchase Behavior Dataset (Kaggle)

Features:
- Customer_ID, Age, Gender, City, Category
- Purchase_Amount, Payment_Method, Discount_Applied, Rating
- Purchase_Date, Repeat_Customer
