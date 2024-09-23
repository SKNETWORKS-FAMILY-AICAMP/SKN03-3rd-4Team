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

    return render(request, 'admin_analysis.html', {
        'gender_male': gender_male,
        'gender_female': gender_female,
    })



