# SKN03-3rd-4Team - 4ævis

---

## 프로젝트 개요

이 프로젝트는 **RandomForestClassifier**를 사용한 분류 모델을 학습하고 평가하는 데 중점을 두고 있습니다. **SMOTE**를 통해 클래스 불균형 문제를 처리하고, **GridSearchCV**로 하이퍼파라미터 튜닝을 수행하며, **ROC-AUC** 평가와 **SHAP** 분석을 통해 모델의 성능과 설명력을 확인합니다. 주요 목표는 성능이 좋은 분류 모델을 구축하고, 그 결과를 설명할 수 있는 모델을 만드는 것입니다.

---

### 요구 사항

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
    - `shap`
    - `joblib`

```bash
pip install -r requirements.txt

```

---

## 주요 파일 설명

### 1. **`ml_models.py`**

- `train_random_forest`: **RandomForestClassifier**를 사용하여 학습하는 함수로, **preprocessor**, **SMOTE**를 포함한 파이프라인을 구성하고 **GridSearchCV**를 통해 최적의 하이퍼파라미터를 찾습니다.
- `evaluate_model`: 학습된 모델을 평가하는 함수로, **classification report**, **ROC-AUC** 스코어를 출력하고 **특성 중요도(Feature Importance)** 및 **ROC 커브**를 시각화합니다.
- `save_ml_model`: 학습된 모델을 파일로 저장하는 함수입니다.

### 2. **`utils.py`**

- 모델 학습과 평가에 필요한 유틸리티 함수들을 포함하고 있습니다.
    - `save_model`: 모델을 저장하는 함수.
    - `plot_roc_curve`: ROC 커브를 시각화하는 함수.
    - `plot_feature_importance`: 모델의 특성 중요도를 시각화하는 함수.
    - `shap_analysis`: **SHAP** 분석을 통해 모델의 예측 결과를 설명하는 함수.

---

## 모델 학습 및 평가

1. **RandomForest 모델 학습**
    
    `train_random_forest` 함수는 데이터 전처리, SMOTE를 통한 불균형 처리, **GridSearchCV**를 사용한 하이퍼파라미터 튜닝 과정을 포함한 파이프라인을 구성하여 **RandomForestClassifier** 모델을 학습합니다.
    
    ```python
    best_model = train_random_forest(X_train, y_train, preprocessor)
    
    ```
    
2. **모델 평가**
    
    `evaluate_model` 함수를 사용하여 모델의 성능을 평가하고, ROC-AUC 점수 및 특성 중요도를 시각화할 수 있습니다. 또한, **SHAP 분석**을 통해 모델의 예측 결과를 해석할 수 있습니다.
    
    ```python
    evaluate_model(best_model, X_val, y_val)
    
    ```
    
3. **모델 저장**
    
    학습이 완료된 모델은 `save_ml_model`을 사용해 로컬에 저장할 수 있습니다.
    
    ```python
    save_ml_model(best_model, 'models/best_rf_model.pkl')
    
    ```
    

---

## 참고 사항

- SHAP 분석을 진행할 때 희소 행렬(sparse matrix)을 밀집 행렬로 변환하는 과정이 필요할 수 있습니다.
- 모델의 하이퍼파라미터 튜닝 결과는 `GridSearchCV`를 통해 확인할 수 있으며, 최적의 파라미터가 출력됩니다.

```bash
최적 하이퍼파라미터: {'classifier__n_estimators': 300, 'classifier__max_depth': None, 'classifier__min_samples_split': 5}

```

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
