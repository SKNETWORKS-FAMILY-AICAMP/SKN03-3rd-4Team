import numpy as np
import pandas as pd
import os

def preprocess_input(input_data):
    
    dict = {}


    dict["customer_id"] = '0000-ABCD'
    dict["seniorcitizen_2017"] = input_data['seniorCitizen']
    dict["partner_2017"] = input_data['partner']
    dict["dependents_2017"] = input_data['dependents']
    dict["tenure_2017"] = input_data['tenure']
    dict["phoneservice_2017"] = input_data["phoneService"]
    dict["multiplelines_2017"] = input_data["multipleLines"]
    dict["internetservice_2017"] = input_data["internetService"]
    dict["onlinesecurity_2017"] = input_data["onlineSecurity"]
    dict["onlinebackup_2017"] = input_data["onlineBackup"]
    dict["deviceprotection_2017"] = input_data["deviceProtection"]
    dict["techsupport_2017"] = input_data["techSupport"]
    dict["streamingtv_2017"] = input_data["streamingTV"]
    dict["streamingmovies_2017"] = input_data["streamingMovies"]
    dict["contract_2017"] = input_data["contract"]
    dict["paperlessbilling_2017"] = input_data["paperlessBilling"]
    dict["paymentmethod_2017"] = input_data["paymentMethod"]
    dict["monthlycharges_2017"] = input_data["monthlyCharges"]
    dict["totalcharges_2017"] = input_data["totalCharges"]
    dict["churn_2017"] = "No"
    dict["seniorcitizen_2018"] = input_data['seniorCitizen']
    dict["partner_2018"] = input_data['partner']
    dict["dependents_2018"] = input_data['dependents']
    dict["tenure_2018"] = input_data['tenure']
    dict["phoneservice_2018"] = input_data["phoneService"]
    dict["multiplelines_2018"] = input_data["multipleLines"]
    dict["internetservice_2018"] = input_data["internetService"]
    dict["onlinesecurity_2018"] = input_data["onlineSecurity"]
    dict["onlinebackup_2018"] = input_data["onlineBackup"]
    dict["deviceprotection_2018"] = input_data["deviceProtection"]
    dict["techsupport_2018"] = input_data["techSupport"]
    dict["streamingtv_2018"] = input_data["streamingTV"]
    dict["streamingmovies_2018"] = input_data["streamingMovies"]
    dict["contract_2018"] = input_data["contract"]
    dict["paperlessbilling_2018"] = input_data["paperlessBilling"]
    dict["paymentmethod_2018"] = input_data["paymentMethod"]
    dict["monthlycharges_2018"] = input_data["monthlyCharges"]
    dict["totalcharges_2018"] = input_data["totalCharges"]
    dict["churn_2018"] = "No"
    dict["seniorcitizen_2019"] = input_data['seniorCitizen']
    dict["partner_2019"] = 1 if input_data['partner'] == 'Yes' else 0
    dict["dependents_2019"] = input_data['dependents']
    dict["tenure_2019"] = input_data['tenure']
    dict["phoneservice_2019"] = 1 if input_data["phoneService"] == 'Yes' else 0
    dict["multiplelines_2019"] = 0 if input_data["multipleLines"] != 'Yes' else 1
    dict["internetservice_2019"] = 0 if  input_data["internetService"] != 'Yes'else 1
    dict["onlinesecurity_2019"] = 0 if  input_data["onlineSecurity"] != 'Yes'else 1
    dict["onlinebackup_2019"] = 0 if  input_data["onlineBackup"] != 'Yes'else 1
    dict["deviceprotection_2019"] = 0 if  input_data["deviceProtection"] != 'Yes' else 1
    dict["techsupport_2019"] = 0 if  input_data["techSupport"] != 'Yes' else 1
    dict["streamingtv_2019"] = 0 if  input_data["streamingTV"] != 'Yes' else 1
    dict["streamingmovies_2019"] = 0 if  input_data["streamingMovies"] != 'Yes' else 1
    dict["contract_2019"] = 0
    dict["paperlessbilling_2019"] = 0
    dict["paymentmethod_2019"] = 0
    dict["monthlycharges_2019"] = input_data["monthlyCharges"]
    dict["totalcharges_2019"] = input_data["totalCharges"]
    dict["churn_2019"] = 'No'
    dict["seniorcitizen_2020"] = input_data['seniorCitizen']
    dict["partner_2020"] = input_data['partner']
    dict["dependents_2020"] = input_data['dependents']
    dict["tenure_2020"] = input_data['tenure']
    dict["phoneservice_2020"] = input_data["phoneService"]
    dict["multiplelines_2020"] = input_data["multipleLines"]
    dict["internetservice_2020"] = input_data["internetService"]
    dict["onlinesecurity_2020"] = input_data["onlineSecurity"]
    dict["onlinebackup_2020"] = input_data["onlineBackup"]
    dict["deviceprotection_2020"] = input_data["deviceProtection"]
    dict["techsupport_2020"] = input_data["techSupport"]
    dict["streamingtv_2020"] = input_data["streamingTV"]
    dict["streamingmovies_2020"] = input_data["streamingMovies"]
    dict["contract_2020"] = input_data["contract"]
    dict["paperlessbilling_2020"] = input_data["paperlessBilling"]
    dict["paymentmethod_2020"] = input_data["paymentMethod"]
    dict["monthlycharges_2020"] = input_data["monthlyCharges"]
    dict["totalcharges_2020"] = input_data["totalCharges"]
    dict["churn_2020"] = "No"
    dict["age_2022"] = 21
    dict["married_2022"] = "No"
    dict["number_of_dependents_2022"] = input_data['dependents']
    dict["city_2022"] = "Los Angeles"
    dict["zip_code_2022"] = 90001
    dict["latitude_2022"] = 33.973616
    dict["longitude_2022"] = -118.24902
    dict["number_of_referrals_2022"] = 0
    dict["tenure_in_months_2022"] = input_data['tenure']
    dict["offer_2022"] = "Offer E"
    dict["phone_service_2022"] = input_data["phoneService"]
    dict["multiple_lines_2022"] = input_data["multipleLines"]
    dict["internet_service_2022"] = "Yes" if input_data["internetService"] in ['DSL', 'Fiber optic'] else "No"
    dict["internet_type_2022"] = input_data["internetService"]
    dict["online_security_2022"] = input_data["onlineSecurity"]
    dict["online_backup_2022"] = input_data["onlineBackup"]
    dict["device_protection_plan_2022"] = input_data["deviceProtection"]
    dict["premium_tech_support_2022"] = input_data["techSupport"]
    dict["streaming_tv_2022"] = input_data["streamingTV"]
    dict["streaming_movies_2022"] = input_data["streamingMovies"]
    dict["contract_2022"] = input_data["contract"]
    dict["paperless_billing_2022"] = input_data["paperlessBilling"]
    dict["payment_method_2022"] = input_data["paymentMethod"]
    dict["monthly_charge_2022"] = input_data["monthlyCharges"]
    dict["total_charges_2022"] = input_data["totalCharges"]
    dict["total_refunds_2022"] =0 
    dict["total_extra_data_charges_2022"] = 0.0
    dict["total_long_distance_charges_2022"] = 0
    dict["total_revenue_2022"] = input_data["totalCharges"]
    dict["customer_status_2022"] = 'Joined'
    dict["churn_2022"] = 'No'
    dict["senior_citizen_2023"] = input_data['seniorCitizen']
    dict["partner_2023"] = input_data['partner']
    dict["dependents_2023"] = input_data['dependents']
    dict["tenure_2023"] = input_data['tenure']
    dict["phone_service_2023"] = input_data["phoneService"]
    dict["multiple_lines_2023"] = input_data["multipleLines"]
    dict["internet_service_2023"] = input_data["internetService"]
    dict["online_security_2023"] = input_data["onlineSecurity"]
    dict["online_backup_2023"] = input_data["onlineBackup"]
    dict["device_protection_2023"] = input_data["deviceProtection"]
    dict["tech_support_2023"] = input_data["techSupport"]
    dict["streaming_tv_2023"] = input_data["streamingTV"]
    dict["streaming_movies_2023"] = input_data["streamingMovies"]
    dict["contract_2023"] = input_data["contract"]
    dict["paperless_billing_2023"] = input_data["paperlessBilling"]
    dict["payment_method_2023"] = input_data["paymentMethod"]
    dict["monthly_charges_2023"] = input_data["monthlyCharges"]
    dict["total_charges_2023"] = input_data["totalCharges"]
    dict["churn_2023"] = "No"
    dict["gender"] = input_data["gender"]


    print('?')
    print(type(dict))
    print(dict)
    
    df = pd.DataFrame([dict])
    print(df)
    # 'tenure' 값을 실수로 변환
    df['tenure_2017'] = df['tenure_2017'].astype(float)
    df['tenure_2018'] = df['tenure_2018'].astype(float)
    df['tenure_2019'] = df['tenure_2019'].astype(float)
    df['tenure_2020'] = df['tenure_2020'].astype(float)
    df['tenure_in_months_2022'] = df['tenure_in_months_2022'].astype(float)
    df['tenure_2023'] = df['tenure_2023'].astype(float)

    df['monthlycharges_2017'] = df['monthlycharges_2017'].astype(float)
    df['monthlycharges_2018'] = df['monthlycharges_2018'].astype(float)
    df['monthlycharges_2019'] = df['monthlycharges_2019'].astype(float)
    df['monthlycharges_2020'] = df['monthlycharges_2020'].astype(float)
    df['monthly_charge_2022'] = df['monthly_charge_2022'].astype(float)
    df['monthly_charges_2023'] = df['monthly_charges_2023'].astype(float)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = 'Churn_Data_-_Final_Version.csv'
    file_path = os.path.join(BASE_DIR, 'data', file_path)
    df_csv = pd.read_csv(file_path)

    if 'Unnamed: 0' in df_csv.columns:
        df_csv = df_csv.drop(columns=['Unnamed: 0'])

    df_csv.iloc[0] = df.iloc[0]
    df_csv.to_csv(file_path, index=True)

    # 추가적인 전처리 단계 (정규화, 스케일링 등)가 필요하면 여기서 수행
    return df_csv
