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


<img width="817" alt="스크린샷 2024-09-24 오전 11 16 21" src="https://github.com/user-attachments/assets/ede2a37e-f139-4809-b9be-0652d6beb322">


### 공통 유틸리티

#### `utils.py`

- 모델 저장, ROC 커브 시각화, 특성 중요도 시각화, **SHAP** 분석 등의 공통 기능을 제공합니다.


---

## 오류 해결 과정

프로젝트 진행 중 다음과 같은 오류 및 문제를 해결했습니다:
1. 모델 저장 및 불러오기 시 오류

	•	오류 원인: focal_loss 함수가 Keras에서 직렬화되지 않음.
	•	해결 방법: @tf.keras.utils.register_keras_serializable() 데코레이터를 추가하여 Keras에서 직렬화 가능하게 함.

2. 모델 예측 후 결과 처리 오류

	•	오류 원인: 예측값이 리스트 형태로 반환되고, 차원 불일치로 인해 numpy.concatenate() 함수 사용 시 오류 발생.
	•	해결 방법: 각 예측값을 1차원 배열로 변환하여 np.concatenate()를 사용함.

3. Attention 레이어 관련 오류

	•	오류 원인: 모델이 호출되지 않은 상태에서 Attention 가중치를 추출하려고 함.
	•	해결 방법: model.predict()를 먼저 호출해 모델을 실행한 후, Attention 가중치를 추출하도록 변경함.

4. 데이터 전처리 관련 오류

	•	오류 원인: year 컬럼이 존재하지 않아 데이터 시계열 정렬 중 KeyError 발생.
	•	해결 방법: 전처리 단계에서 시계열 컬럼을 올바르게 생성하도록 수정함.

5. 예측 시 차원 불일치 오류

	•	오류 원인: 예측값의 차원이 다름.
	•	해결 방법: 예측값을 평탄화하고 일관된 차원으로 맞춤.


### 결과 분석


#### 1. **Random Forest 결과**
![rf_rocauc_curve](https://github.com/user-attachments/assets/fc24e0cd-da5b-4b34-b1d8-bedc19f356e6)



##### Random Forest 모델 성능

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.88      | 0.85   | 0.87     | 931     |
| 1     | 0.62      | 0.69   | 0.65     | 337     |

| Accuracy   | 0.81 |
| Macro avg  | 0.75 |
| Weighted avg | 0.81 |

**Random Forest ROC-AUC Score**: 0.8614

- 주요 변수: 계약 유형, 월 요금, 가입 기간

![rf_feature_importance](https://github.com/user-attachments/assets/9bc211b1-9d90-4e14-8b40-422eb78e7d9e)

---

#### 2. **LSTM 결과**

![lstm_auc_graph](https://github.com/user-attachments/assets/fc26a240-8913-4b83-9f97-a6746f0f2225)
![lstm_loss_graph](https://github.com/user-attachments/assets/72834889-bbd7-459f-a214-eb6cd79b0fa4)

##### LSTM 모델 성능

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0.0   | 1.00      | 0.99   | 1.00     | 1033    |
| 1.0   | 0.98      | 1.00   | 0.99     | 376     |

| Accuracy   | 0.99 |
| Macro avg  | 0.99 |
| Weighted avg | 0.99 |

**LSTM ROC-AUC Score**: 1.0
- 어텐션 메커니즘을 통해 중요한 시점에 집중하여 더 높은 성능을 보임.

![lstm_confusion_matrix](https://github.com/user-attachments/assets/6b8bc69d-be69-4b91-a7f2-8065a40dd63b)

---

#### 3. **Attention 가중치 시각화**
![attention_weight_value](https://github.com/user-attachments/assets/fa0caf86-0b0a-4f3d-b4b1-579584eacf6c)

- 각 시점별 가중치가 균등하게 분포하여 장기적인 패턴 학습에 도움이 되었음.


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
