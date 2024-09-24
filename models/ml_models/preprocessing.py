import numpy as np
import pandas as pd

def preprocess_input(data):
    years = [2017, 2018, 2019, 2020, 2022, 2023]
    expanded_data = {}
    expanded_data['customer_id'] = '0000-OOOOO'
    
    for year in years:
        if year in [2017, 2018, 2019, 2020, 2023]:
            expanded_data[f'seniorcitizen_{year}'] = data['seniorCitizen']
            expanded_data[f'partner_{year}'] = data['partner']
            expanded_data[f'dependents_{year}'] = data['dependents']
            expanded_data[f'tenure_{year}'] = float(data['tenure'])
            expanded_data[f'phoneservice_{year}'] = data['phoneService']
            expanded_data[f'multiplelines_{year}'] = data['multipleLines']
            expanded_data[f'internetservice_{year}'] = data['internetService']
            expanded_data[f'onlinesecurity_{year}'] = data['onlineSecurity']
            expanded_data[f'onlinebackup_{year}'] = data['onlineBackup']
            expanded_data[f'deviceprotection_{year}'] = data['deviceProtection']
            expanded_data[f'techsupport_{year}'] = data['techSupport']
            expanded_data[f'streamingtv_{year}'] = data['streamingTV']
            expanded_data[f'streamingmovies_{year}'] = data['streamingMovies']
            expanded_data[f'contract_{year}'] = data['contract']
            expanded_data[f'paperlessbilling_{year}'] = data['paperlessBilling']
            expanded_data[f'paymentmethod_{year}'] = data['paymentMethod']
            expanded_data[f'monthlycharges_{year}'] = float(data['monthlyCharges']) if data['monthlyCharges'] else np.nan
            expanded_data[f'totalcharges_{year}'] = float(data['totalCharges']) if data['totalCharges'] else np.nan
            expanded_data[f'churn_{year}'] = 'No'  # 예시로 None을 넣었지만, 실제 데이터에 맞게 설정 필요

    # 2022년에 대해서는 추가적인 컬럼 처리
    expanded_data['age_2022'] = data.get('age', 21)
    expanded_data['married_2022'] = data.get('married', 'No')
    expanded_data['number_of_dependents_2022'] = data.get('number_of_dependents', 0)
    expanded_data['city_2022'] = data.get('city', 'Los Angeles')
    expanded_data['zip_code_2022'] = data.get('zip_code', 90013)
    expanded_data['latitude_2022'] = data.get('latitude', 34.044639)
    expanded_data['longitude_2022'] = data.get('longitude', -118.240413)
    expanded_data['number_of_referrals_2022'] = data.get('number_of_referrals', 0)
    expanded_data['tenure_in_months_2022'] = data.get('tenure_in_months', 'Offer D')
    expanded_data['offer_2022'] = data.get('offer', 'Yes')
    expanded_data['phone_service_2022'] = data.get('phone_service', 'No')
    expanded_data['multiple_lines_2022'] = data.get('multiple_lines', 'No')
    expanded_data['internet_service_2022'] = data.get('internet_service', 'No')
    expanded_data['internet_type_2022'] = data.get('internet_type', 'No')
    expanded_data['online_security_2022'] = data.get('online_security', 'No internet service')
    expanded_data['online_backup_2022'] = data.get('online_backup', 'No internet service')
    expanded_data['device_protection_plan_2022'] = data.get('device_protection_plan', 'No internet service')
    expanded_data['premium_tech_support_2022'] = data.get('premium_tech_support', 'No internet service')
    expanded_data['streaming_tv_2022'] = data.get('streaming_tv', 'No internet service')
    expanded_data['streaming_movies_2022'] = data.get('streaming_movies', 'No internet service')
    expanded_data['contract_2022'] = data.get('contract', 'One Year')
    expanded_data['paperless_billing_2022'] = data.get('paperless_billing', 'No')
    expanded_data['payment_method_2022'] = data.get('payment_method', 'Mailed Check')
    expanded_data['monthly_charge_2022'] = float(data.get('monthly_charge', 20.65))
    expanded_data['total_charges_2022'] = float(data.get('total_charges', 1022.95))
    expanded_data['total_refunds_2022'] = float(data.get('total_refunds', 0.0))
    expanded_data['total_extra_data_charges_2022'] = float(data.get('total_extra_data_charges', 0))
    expanded_data['total_long_distance_charges_2022'] = float(data.get('total_long_distance_charges', 1172.6))
    expanded_data['total_revenue_2022'] = float(data.get('total_revenue', 2195.55))
    expanded_data['customer_status_2022'] = data.get('customer_status', 'Stayed')
    expanded_data['churn_2022'] = None  # 예시로 None을 넣었지만, 실제 데이터에 맞게 설정 필요

    # gender는 별도로 관리되므로 포함하지 않음
    expanded_data['gender'] = data['gender']
    
    print('예측?')
    print(expanded_data)
    df = pd.DataFrame([expanded_data])
    df.to_csv('expanded_data.csv', index=False)
    
    return df
