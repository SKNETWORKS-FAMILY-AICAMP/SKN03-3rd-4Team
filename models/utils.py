# src/utils.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier  # XGBoost 분류 모델
from sklearn.ensemble import RandomForestClassifier  # RandomForest 분류 모델
import joblib
import shap
import matplotlib.pyplot as plt

def load_data(file_path):
    """데이터 로드 및 기본 전처리"""
    data = pd.read_csv(file_path)
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    if 'offer_2022' in data.columns:
        data = data.drop(columns=['offer_2022'])
    return data

def rename_columns(data):
    """열 이름 통일"""
    rename_mapping = {
        'monthly_charges_2023': 'monthlycharges_2023',
        'total_charges_2023': 'totalcharges_2023',
        'senior_citizen_2023': 'seniorcitizen_2023',
        'phone_service_2023': 'phoneservice_2023',
        'multiple_lines_2023': 'multiplelines_2023',
        'internet_service_2023': 'internetservice_2023',
        'online_security_2023': 'onlinesecurity_2023',
        'online_backup_2023': 'onlinebackup_2023',
        'device_protection_2023': 'deviceprotection_2023',
        'tech_support_2023': 'techsupport_2023',
        'streaming_tv_2023': 'streamingtv_2023',
        'streaming_movies_2023': 'streamingmovies_2023',
        'contract_2023': 'contract_2023',
        'paperless_billing_2023': 'paperlessbilling_2023',
        'payment_method_2023': 'paymentmethod_2023'
    }
    data.rename(columns=rename_mapping, inplace=True)
    return data

def handle_missing_values(data, columns):
    """빈 문자열을 NaN으로 변환하고 최빈값으로 대체"""
    data.replace(' ', np.nan, inplace=True)
    for col in columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            mode_value = data[col].mode()[0]
            data[col].fillna(mode_value, inplace=True)
    return data

def encode_target_variable(data):
    """목표 변수 인코딩"""
    if 'churn_2023' in data.columns:
        data['churn'] = data['churn_2023'].apply(lambda x: 1 if x == 'Yes' else 0)
    else:
        raise ValueError("Column 'churn_2023' not found in the data.")
    churn_cols = [col for col in data.columns if 'churn' in col and col != 'churn']
    data = data.drop(columns=churn_cols)
    return data

def feature_engineering(data):
    """피처 엔지니어링"""
    data['tenure_change_2017_2023'] = data['tenure_2023'] - data['tenure_2017']
    data['monthly_charges_change_2017_2023'] = data['monthlycharges_2023'] - data['monthlycharges_2017']
    data['internet_and_tv'] = ((data['internetservice_2023'] != 'No') & (data['streamingtv_2023'] == 'Yes')).astype(int)
    data['has_tech_support_bundle'] = (
        (data['techsupport_2023'] == 'Yes') &
        (data['onlinesecurity_2023'] == 'Yes') &
        (data['onlinebackup_2023'] == 'Yes')
    ).astype(int)
    data['internet_service_change'] = (data['internetservice_2023'] != data['internetservice_2017']).astype(int)
    data['payment_method_change'] = (data['paymentmethod_2023'] != data['paymentmethod_2017']).astype(int)
    data['long_term_contract'] = (data['contract_2023'] == 'Two year').astype(int)
    data['has_online_security'] = (data['onlinesecurity_2023'] == 'Yes').astype(int)
    data['has_online_backup'] = (data['onlinebackup_2023'] == 'Yes').astype(int)
    return data

def preprocess_data_ml(data):
    """머신러닝 모델을 위한 데이터 전처리"""
    categorical_features = [col for col in data.columns if 'contract' in col or 'payment' in col or 'gender' in col or 'city' in col]
    numerical_features = [col for col in data.columns if 'charges' in col or 'tenure' in col or col.startswith('seniorcitizen') or col.startswith('dependents')]
    
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X = data.drop(columns=['churn', 'customer_id'])
    y = data['churn']
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, preprocessor

def save_model(model, filepath):
    """모델 저장"""
    joblib.dump(model, filepath)
    print(f"모델이 {filepath}에 저장되었습니다.")

def plot_roc_curve(y_true, y_proba, model_name):
    """ROC-AUC 커브 시각화"""
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8,6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC-AUC Curve')
    plt.legend(loc='lower right')
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20):
    """특징 중요도 시각화"""
    importances = model.named_steps['classifier'].feature_importances_
    indices = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10,8))
    plt.title('Feature Importances')
    plt.barh(range(len(indices)), importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.show()

def shap_analysis(model, X):
    """SHAP 분석"""
    # 모델이 트리 기반 모델이라면 TreeExplainer를 사용
    if isinstance(model.named_steps['classifier'], (XGBClassifier, RandomForestClassifier)):
        explainer = shap.TreeExplainer(model.named_steps['classifier'])
    else:
        explainer = shap.Explainer(model.named_steps['classifier'])

    # 데이터가 클 경우 일부 샘플링
    if X.shape[0] > 500:  # 예: 샘플이 500개 이상일 경우 샘플링
        X_sampled = shap.sample(X, 500, random_state=42)  # 500개의 샘플을 랜덤하게 선택
    else:
        X_sampled = X

    shap_values = explainer(X_sampled)
    shap.summary_plot(shap_values, X_sampled, plot_type="bar")