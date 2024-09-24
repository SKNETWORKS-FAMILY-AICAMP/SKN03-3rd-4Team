from django.shortcuts import render
from django.http import JsonResponse
import json
import joblib
import os
from models.ml_models.preprocessing import preprocess_input
from tensorflow.keras.models import load_model
from models.data_preprocessing import preprocessing_ml
from models.data_preprocessing import preprocess
from models.lstm_model import preprocess_lstm
from models.lstm_model import focal_loss_fixed

# Create your views here.
def prediction(request):
    
    return render(request, 'prediction.html')

def predict_churn(request):
    data = json.loads(request.body)
    
    
    dl_input_data = preprocess_input(data)
    print('ddddd')
    print(dl_input_data)
















    # 현재 파일의 경로를 기준으로 상대 경로를 계산
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ml_model_path = os.path.join(BASE_DIR, 'models/ml_models', 'best_rf_model.pkl')

    # X_train_ml, X_val_ml, X_test_ml, y_train_ml, y_val_ml, y_test_ml, preprocessor, data_ml = preprocess()


    # 업로드 된 머신러닝 모델 불러오기
    # ml_model = joblib.load(ml_model_path)
    # # 데이터 전처리
    # ml_input_data = preprocessing_ml(data)
    # # 예측
    # prediction = ml_model.predict(X_test_ml)
    # print(prediction)
    


    X_test_lstm, y_test_lstm, n_features, max_seq_length, class_weight_dict = preprocess_lstm(dl_input_data)
    # 업로드 된 딥모델 파일 경로

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ml_model_path = os.path.join(BASE_DIR, 'models/dl_models', 'lstm_model.keras')

    print('모델로드')
    # 모델 로드
    dl_model = load_model(ml_model_path, custom_objects={'focal_loss_fixed': focal_loss_fixed})
    # 예측
    print('예측시작')
    predictions = dl_model.predict(X_test_lstm).flatten()
    print('예측종료')
    print(type(X_test_lstm))

    return JsonResponse({'result': predictions.tolist() })