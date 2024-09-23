from django.shortcuts import render
from django.http import JsonResponse
import json
import joblib
from ml_models.preprocessing import preprocess_input

# Create your views here.
def prediction(request):
    
    return render(request, 'prediction.html')

def predict_churn(request):
    data = json.loads(request.body)
    
    # input_data = preprocess_input(data)

    # 모델 불러오기
    # model = joblib.load('ml_model/VotingClassifier_best_model.pkl')
    # 예측
    # prediction = model.predict(input_data)

    return JsonResponse({'result': 'ㅇ' })