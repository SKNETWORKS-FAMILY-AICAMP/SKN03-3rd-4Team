from django.shortcuts import render
from django.http import JsonResponse
import json
import joblib
import os
from ml_models.preprocessing import preprocess_input
# from tensorflow.keras.models import load_model

# Create your views here.
def prediction(request):
    
    return render(request, 'prediction.html')

def predict_churn(request):
    data = json.loads(request.body)
    
    # ml_input_data = preprocess_input(data)
    # dl_input_data = preprocess_input(data)

    # 현재 파일의 경로를 기준으로 상대 경로를 계산
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ml_model_path = os.path.join(BASE_DIR, 'ml_models', 'best_rf_model.pkl')


    # 업로드 된 머신러닝 모델 불러오기
    ml_model = joblib.load(ml_model_path)

    # 예측
    # prediction = ml_model.predict(ml_input_data)
    
    # 업로드 된 딥모델 파일 경로
    # 모델 로드
    # dl_model = load_model('media/models/your_model.h5')
    # 예측
    # predictions = dl_model.predict(dl_input_data)


    return JsonResponse({'result': 'ㅇ' })