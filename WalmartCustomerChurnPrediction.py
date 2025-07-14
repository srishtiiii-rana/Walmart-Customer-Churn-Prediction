import streamlit as st
import kagglehub
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Walmart Customer Churn Prediction",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🛒 Walmart Customer Churn Prediction</h1>', unsafe_allow_html=True)
st.markdown("---")

# Load dataset function
@st.cache_data
def load_kaggle_data():
    try:
        # Download latest version
        path = kagglehub.dataset_download("logiccraftbyhimanshi/walmart-customer-purchase-behavior-dataset")
        st.success(f"Dataset downloaded successfully!")
        
        # Load the data
        files = os.listdir(path)
        csv_files = [f for f in files if f.endswith('.csv')]
        
        if csv_files:
            df = pd.read_csv(os.path.join(path, csv_files[0]))
            return df
        else:
            st.error("No CSV files found in the dataset")
            return None
    except Exception as e:
        st.error(f"Error loading Kaggle dataset: {str(e)}")
        return None

# Data preprocessing function
def preprocess_data(df):
    """Preprocess the data for churn prediction"""
    try:
        df_processed = df.copy()
        
        # Convert Purchase_Date to datetime
        df_processed['Purchase_Date'] = pd.to_datetime(df_processed['Purchase_Date'])
        
        # Create recency-based churn target
        latest_date = df_processed['Purchase_Date'].max()
        df_processed['Days_Since_Last_Purchase'] = (latest_date - df_processed['Purchase_Date']).dt.days
        df_processed['Churn_Recency'] = (df_processed['Days_Since_Last_Purchase'] > 90).astype(int)
        
        # Create repeat customer churn target
        df_processed['Churn_Repeat'] = (df_processed['Repeat_Customer'] == 'No').astype(int)
        
        return df_processed
        
    except Exception as e:
        st.error(f"Error in data preprocessing: {str(e)}")
        return None

# Model training function
def train_models(df, target_col, features):
    """Train multiple models and return the best one"""
    try:
        # Filter available features
        available_features = [col for col in features if col in df.columns]
        
        if not available_features:
            st.error("No features available for training")
            return None, None, None, None
        
        # Encode categorical variables
        df_encoded = df.copy()
        le_dict = {}
        
        for col in available_features:
            if df_encoded[col].dtype == 'object':
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                le_dict[col] = le
        
        X = df_encoded[available_features]
        y = df_encoded[target_col]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        model_results = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            model_results[name] = {
                'model': model,
                'accuracy': accuracy,
                'predictions': y_pred,
                'y_test': y_test
            }
        
        # Return the best model
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['accuracy'])
        return model_results[best_model_name], model_results, le_dict, available_features
        
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return None, None, None, None

# Predict churn probabilities
def predict_churn_risk(df, model, le_dict, features):
    """Predict churn risk for all customers"""
    try:
        df_encoded = df.copy()
        
        # Encode categorical variables using the same encoders
        for col in features:
            if col in le_dict:
                df_encoded[col] = le_dict[col].transform(df_encoded[col].astype(str))
        
        # Get predictions
        X = df_encoded[features]
        probabilities = model.predict_proba(X)[:, 1]  # Probability of churn
        
        return probabilities
    except Exception as e:
        st.error(f"Error in risk prediction: {str(e)}")
        return None

# Main app logic
def main():
    # Sidebar for data source selection
    st.sidebar.header("📊 Data Source")
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV File", "Use Kaggle Dataset"]
    )
    
    df = None
    
    if data_source == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="Upload a CSV file with customer data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.sidebar.success("✅ File uploaded successfully!")
            except Exception as e:
                st.sidebar.error(f"Error reading file: {str(e)}")
    
    else:
        with st.spinner("Loading Kaggle dataset..."):
            df = load_kaggle_data()
    
    if df is not None:
        # Display dataset info
        st.subheader("📋 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Total Customers", df['Customer_ID'].nunique())
        with col3:
            st.metric("Date Range", f"{df['Purchase_Date'].min()} to {df['Purchase_Date'].max()}")
        with col4:
            st.metric("Avg Purchase Amount", f"${df['Purchase_Amount'].mean():.2f}")
        
        # Preprocess data
        df_processed = preprocess_data(df)
        
        if df_processed is None:
            st.error("❌ Data preprocessing failed. Please check your data format.")
            return
        
        # Features for modeling
        features = ['Age', 'Gender', 'City', 'Category', 'Purchase_Amount', 'Payment_Method', 'Discount_Applied', 'Rating']
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["🎯 Repeat Customer Analysis", "⏰ Recency Analysis", "🚨 Combined Risk Analysis"])
        
        with tab1:
            st.subheader("🎯 Repeat Customer Churn Analysis")
            
            # Train model for repeat customer churn
            with st.spinner("Training model for repeat customer analysis..."):
                best_model, all_models, le_dict_repeat, model_features = train_models(df_processed, 'Churn_Repeat', features)
                
                if best_model is None:
                    st.error("❌ Model training failed")
                    return
            
            # Display model performance
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Model Accuracy", f"{best_model['accuracy']:.3f}")
                
                # Model comparison
                st.subheader("Model Comparison")
                model_comparison = pd.DataFrame({
                    'Model': list(all_models.keys()),
                    'Accuracy': [all_models[model]['accuracy'] for model in all_models.keys()]
                })
                st.dataframe(model_comparison)
            
            with col2:
                # Confusion matrix
                cm = confusion_matrix(best_model['y_test'], best_model['predictions'])
                fig = px.imshow(cm, text_auto=True, title='Confusion Matrix')
                st.plotly_chart(fig, use_container_width=True)
            
            # Predict churn risk for all customers
            churn_probabilities = predict_churn_risk(df_processed, best_model['model'], le_dict_repeat, model_features)
            
            if churn_probabilities is not None:
                df_processed['Churn_Risk_Repeat'] = churn_probabilities
                
                # Top risk customers
                st.subheader("Top 100 At-Risk Customers (Repeat Customer Analysis)")
                top_risk_repeat = df_processed.nlargest(100, 'Churn_Risk_Repeat')[['Customer_ID', 'Churn_Risk_Repeat']].copy()
                top_risk_repeat['Risk_Percentage'] = (top_risk_repeat['Churn_Risk_Repeat'] * 100).round(2)
                top_risk_repeat = top_risk_repeat[['Customer_ID', 'Risk_Percentage']].reset_index(drop=True)
                top_risk_repeat.index = top_risk_repeat.index + 1
                
                st.dataframe(top_risk_repeat)
                
                # Download button
                csv = top_risk_repeat.to_csv(index=False)
                st.download_button(
                    label="📥 Download Top 100 Risk Customers (Repeat)",
                    data=csv,
                    file_name='top_100_risk_customers_repeat.csv',
                    mime='text/csv'
                )
        
        with tab2:
            st.subheader("⏰ Recency-Based Churn Analysis")
            
            # Train model for recency-based churn
            with st.spinner("Training model for recency analysis..."):
                best_model_recency, all_models_recency, le_dict_recency, model_features_recency = train_models(df_processed, 'Churn_Recency', features)
                
                if best_model_recency is None:
                    st.error("❌ Model training failed")
                    return
            
            # Display model performance
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Best Model Accuracy", f"{best_model_recency['accuracy']:.3f}")
                
                # Model comparison
                st.subheader("Model Comparison")
                model_comparison_recency = pd.DataFrame({
                    'Model': list(all_models_recency.keys()),
                    'Accuracy': [all_models_recency[model]['accuracy'] for model in all_models_recency.keys()]
                })
                st.dataframe(model_comparison_recency)
            
            with col2:
                # Confusion matrix
                cm_recency = confusion_matrix(best_model_recency['y_test'], best_model_recency['predictions'])
                fig = px.imshow(cm_recency, text_auto=True, title='Confusion Matrix (Recency)')
                st.plotly_chart(fig, use_container_width=True)
            
            # Predict churn risk for all customers
            churn_probabilities_recency = predict_churn_risk(df_processed, best_model_recency['model'], le_dict_recency, model_features_recency)
            
            if churn_probabilities_recency is not None:
                df_processed['Churn_Risk_Recency'] = churn_probabilities_recency
                
                # Top risk customers
                st.subheader("Top 100 At-Risk Customers (Recency Analysis)")
                top_risk_recency = df_processed.nlargest(100, 'Churn_Risk_Recency')[['Customer_ID', 'Churn_Risk_Recency']].copy()
                top_risk_recency['Risk_Percentage'] = (top_risk_recency['Churn_Risk_Recency'] * 100).round(2)
                top_risk_recency = top_risk_recency[['Customer_ID', 'Risk_Percentage']].reset_index(drop=True)
                top_risk_recency.index = top_risk_recency.index + 1
                
                st.dataframe(top_risk_recency)
                
                # Download button
                csv_recency = top_risk_recency.to_csv(index=False)
                st.download_button(
                    label="📥 Download Top 100 Risk Customers (Recency)",
                    data=csv_recency,
                    file_name='top_100_risk_customers_recency.csv',
                    mime='text/csv'
                )
        
        with tab3:
            st.subheader("🚨 Combined Risk Analysis")
            
            # Check if both analyses are available
            if 'Churn_Risk_Repeat' in df_processed.columns and 'Churn_Risk_Recency' in df_processed.columns:
                
                # Determine which approach has higher accuracy
                repeat_accuracy = best_model['accuracy'] if 'best_model' in locals() else 0
                recency_accuracy = best_model_recency['accuracy'] if 'best_model_recency' in locals() else 0
                
                if repeat_accuracy > recency_accuracy:
                    st.success(f"🏆 Best Approach: Repeat Customer Analysis (Accuracy: {repeat_accuracy:.3f})")
                    primary_risk = 'Churn_Risk_Repeat'
                    secondary_risk = 'Churn_Risk_Recency'
                else:
                    st.success(f"🏆 Best Approach: Recency Analysis (Accuracy: {recency_accuracy:.3f})")
                    primary_risk = 'Churn_Risk_Recency'
                    secondary_risk = 'Churn_Risk_Repeat'
                
                # Combine both risk scores (weighted by accuracy)
                total_accuracy = repeat_accuracy + recency_accuracy
                repeat_weight = repeat_accuracy / total_accuracy
                recency_weight = recency_accuracy / total_accuracy
                
                df_processed['Combined_Risk'] = (
                    df_processed['Churn_Risk_Repeat'] * repeat_weight + 
                    df_processed['Churn_Risk_Recency'] * recency_weight
                )
                
                # Display accuracy comparison
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Repeat Customer Accuracy", f"{repeat_accuracy:.3f}")
                with col2:
                    st.metric("Recency Analysis Accuracy", f"{recency_accuracy:.3f}")
                with col3:
                    st.metric("Primary Method", "Repeat" if repeat_accuracy > recency_accuracy else "Recency")
                
                # Top risk customers combined
                st.subheader("Top 100 At-Risk Customers (Combined Analysis)")
                top_risk_combined = df_processed.nlargest(100, 'Combined_Risk')[['Customer_ID', 'Combined_Risk', 'Churn_Risk_Repeat', 'Churn_Risk_Recency']].copy()
                top_risk_combined['Combined_Risk_Percentage'] = (top_risk_combined['Combined_Risk'] * 100).round(2)
                top_risk_combined['Repeat_Risk_Percentage'] = (top_risk_combined['Churn_Risk_Repeat'] * 100).round(2)
                top_risk_combined['Recency_Risk_Percentage'] = (top_risk_combined['Churn_Risk_Recency'] * 100).round(2)
                
                display_df = top_risk_combined[['Customer_ID', 'Combined_Risk_Percentage', 'Repeat_Risk_Percentage', 'Recency_Risk_Percentage']].reset_index(drop=True)
                display_df.index = display_df.index + 1
                
                st.dataframe(display_df)
                
                # Download button
                csv_combined = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Top 100 Risk Customers (Combined)",
                    data=csv_combined,
                    file_name='top_100_risk_customers_combined.csv',
                    mime='text/csv'
                )
                
                # Risk distribution
                fig = px.histogram(df_processed, x='Combined_Risk', title='Combined Risk Score Distribution')
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary stats
                st.subheader("📊 Risk Summary")
                high_risk = len(df_processed[df_processed['Combined_Risk'] > 0.7])
                medium_risk = len(df_processed[(df_processed['Combined_Risk'] > 0.4) & (df_processed['Combined_Risk'] <= 0.7)])
                low_risk = len(df_processed[df_processed['Combined_Risk'] <= 0.4])
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("High Risk (>70%)", high_risk)
                with col2:
                    st.metric("Medium Risk (40-70%)", medium_risk)
                with col3:
                    st.metric("Low Risk (≤40%)", low_risk)
            
            else:
                st.error("❌ Both analyses need to be completed first")
    
    else:
        st.info("Please upload a CSV file or wait for the Kaggle dataset to load.")

if __name__ == "__main__":
    main()