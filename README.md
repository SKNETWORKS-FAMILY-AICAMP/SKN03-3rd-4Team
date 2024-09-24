# SKN03-3rd-4Team - 4ævis

---

## 프로젝트 개요

이 프로젝트는 **RandomForestClassifier**와 **Long Short-Term Memory (LSTM)** 네트워크를 활용하여 고객 이탈(Churn)을 예측하는 두 가지 접근 방식을 비교하고 평가합니다. **SMOTE**를 통해 클래스 불균형 문제를 처리하고, **GridSearchCV**로 하이퍼파라미터 튜닝을 수행하며, **ROC-AUC** 평가와 **SHAP** 분석을 통해 모델의 성능과 설명력을 확인합니다. 주요 목표는 성능이 좋은 분류 모델을 구축하고, 그 결과를 설명할 수 있는 모델을 만드는 것입니다.

---

## 요구 사항

- Python 3.x
- 필수 패키지:
  - `scikit-learn`
  - `imblearn`
  - `scipy`
  - `numpy`
  - `pandas`
  - `keras`
  - `tensorflow`
  - `matplotlib`
  - `seaborn`
  - `shap`
  - `joblib`

```bash
pip install -r requirements.txt
```

---

## 사용 기법 및 알고리즘

### 머신러닝 섹션

#### **RandomForestClassifier**

- **RandomForestClassifier**: 다수의 결정 트리를 사용하여 분류 작업을 수행하는 앙상블 학습 기법입니다. 과적합을 방지하고 높은 예측 성능을 제공합니다.
  
- **SMOTE (Synthetic Minority Over-sampling Technique)**: 소수 클래스의 데이터를 합성하여 클래스 불균형 문제를 해결하는 오버샘플링 기법입니다.
  
- **GridSearchCV**: 하이퍼파라미터 튜닝을 위해 교차 검증을 사용하는 그리드 탐색 방법으로, 최적의 모델 성능을 도출합니다.
  
- **ROC-AUC**: 모델의 분류 성능을 평가하는 지표로, 수신기 조작 특성(ROC) 곡선 아래 면적을 계산하여 성능을 측정합니다.
  
- **SHAP (SHapley Additive exPlanations)**: 모델의 예측 결과를 해석하고, 각 특징이 예측에 미치는 영향을 시각화하는 설명 기법입니다.


#### `lstm_model.py`

#### **Long Short-Term Memory (LSTM)**

- **LSTM**: 시계열 데이터나 순차 데이터를 처리하는 데 강력한 성능을 보이는 순환 신경망(RNN)의 한 종류입니다. 장기 의존성을 학습할 수 있어 고객 이탈 예측에 효과적입니다.
  
- **어텐션 메커니즘**: 모델이 입력 시퀀스의 중요한 부분에 집중할 수 있도록 도와주는 메커니즘으로, 예측 정확도를 향상시킵니다.
  
- **포컬 손실(Focal Loss)**: 클래스 불균형 문제를 해결하기 위해 어려운 샘플에 더 많은 가중치를 부여하는 손실 함수로, 모델이 소수 클래스를 더 잘 학습할 수 있게 합니다.
  
- **시퀀스 패딩**: LSTM 모델의 입력 시퀀스 길이를 통일하기 위해 짧은 시퀀스를 패딩하여 일관된 형태의 데이터를 입력으로 제공합니다.

### 공통 유틸리티

#### `utils.py`

- 모델 저장, ROC 커브 시각화, 특성 중요도 시각화, **SHAP** 분석 등의 공통 기능을 제공합니다.


## 프로젝트 구조

```
churn-prediction/
├── data/
│   └── Churn_Data_-_Final_Version.csv
├── images/
│   └── churn_prediction.png
├── models/
│   ├── best_rf_model.pkl
│   └── lstm_model.keras
├── src/
│   ├── __init__.py
│   ├── ml_models.py
│   ├── lstm_model.py
│   └── utils.py
├── main_ml.py
├── main_dl.py
├── requirements.txt
└── README.md
```

- **data/**: 데이터셋을 포함합니다.
- **images/**: 프로젝트 관련 이미지(예: 다이어그램, 스크린샷)를 저장합니다.
- **models/**: 학습된 모델 파일을 저장합니다.
- **src/**: 소스 코드 모듈을 포함합니다.
  - **ml_models.py**: 머신러닝(RandomForest) 관련 함수들을 구현합니다.
  - **lstm_model.py**: 딥러닝(LSTM) 관련 함수들을 구현합니다.
  - **utils.py**: 공통 유틸리티 함수들을 포함합니다.
- **main_ml.py**: 머신러닝(RandomForest) 모델 전처리 및 학습을 실행하는 메인 스크립트입니다.
- **main_dl.py**: 딥러닝(LSTM) 모델 전처리 및 학습을 실행하는 메인 스크립트입니다.
- **requirements.txt**: 프로젝트 의존성 목록을 포함합니다.
- **README.md**: 프로젝트 문서입니다.

---


## 팀원 소개
| 김병수 | 오승민 | 최연규 | 김종식 | 구나연 | 
|:--:|:--:|:--:|:--:|:--:|
| @BS-KIM-97 | @min3009 | @zelope | @joowon582 | @Leejoowon123 |
## 기술 스택

![html](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![css](https://img.shields.io/badge/CSS-239120?&style=for-the-badge&logo=css3&logoColor=white)
![JS](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=JavaScript&logoColor=white)
![python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)
![Django](https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white)


![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![scipy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![numpy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![matplotlib](https://img.shields.io/badge/Matplotlib-013243?style=for-the-badge&logo=matplotlib&logoColor=white)

