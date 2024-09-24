from django.shortcuts import render
import pandas as pd
import os
import json


def admin_analysis(request):
    file_path = r'C:/dev/github/SKN03-3rd-4Team/Churn_Data_-_Final_Version.csv'
    gender_male = 0
    gender_female = 0
    
    df = pd.read_csv(file_path)  # Load the CSV file
    # 성별 카운트
    if request.method == 'POST':
        df = pd.read_csv(file_path)  # CSV 파일을 불러오기
    for i in range(len(df['gender'])):
        if df['gender'][i]=="Male":
            gender_male+=1
        else:
            gender_female+=1
    churn_2023_yes=0
    chrun_2023_No=0
    for i in range(len(df['churn_2023'])):
        if df['churn_2023'][i]=="Yes":
            churn_2023_yes+=1
        else:
            chrun_2023_No+=1

    churn_yes_tenure=[]
    chrun_no_tenure=[]
    for i in range(len(df['churn_2023'])):
        if df['churn_2023'][i]=="Yes":
            churn_yes_tenure.append(int(df['tenure_2023'][i]))
        else:
            chrun_no_tenure.append(int(df['tenure_2023'][i]))
    yes_tenure_avg=sum(churn_yes_tenure)/len(churn_yes_tenure)
    no_tenure_avg=sum(chrun_no_tenure)/len(chrun_no_tenure)

    return render(request, 'admin_analysis.html', {
        'gender_male': gender_male,
        'gender_female': gender_female,
        'churn_yes':churn_2023_yes,
        'churn_no':chrun_2023_No,
        'yes_tenure_avg':yes_tenure_avg,
        'no_tenure_avg':no_tenure_avg
    })


