# src/lstm_model.py

import pandas as pd
import numpy as np
import os
from .utils import load_data, rename_columns, handle_missing_values, encode_target_variable, feature_engineering, save_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

from tensorflow.keras.saving import register_keras_serializable

def preprocess_lstm(data):
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    file_path = 'Churn_Data_-_Final_Version.csv'
    file_path = os.path.join(BASE_DIR, 'data', file_path)

    df = load_data(file_path)
    # df = data
    df = rename_columns(df)
    df = handle_missing_values(df, ['totalcharges_2017', 'totalcharges_2018', 'totalcharges_2019', 'totalcharges_2020', 'totalcharges_2023'])
    df = encode_target_variable(df)
    df = feature_engineering(df)
    
    # 연도별 데이터 변환
    column_mapping = {
        'number_of_dependents': 'dependents',
        'phone_service': 'phoneservice',
        'multiple_lines': 'multiplelines',
        'internet_service': 'internetservice',
        'internet_type': 'internettype',
        'online_security': 'onlinesecurity',
        'online_backup': 'onlinebackup',
        'device_protection_plan': 'deviceprotection',
        'premium_tech_support': 'techsupport',
        'streaming_tv': 'streamingtv',
        'streaming_movies': 'streamingmovies',
        'paperless_billing': 'paperlessbilling',
        'payment_method': 'paymentmethod',
        'monthly_charge': 'monthlycharges',
        'total_charges': 'totalcharges',
    }
    
    def unify_column_names(df, year, mapping):
        for old_name, new_name in mapping.items():
            old_col = f"{old_name}_{year}"
            new_col = f"{new_name}_{year}"
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        return df
    
    for year in [2017, 2018, 2019, 2020, 2022, 2023]:
        df = unify_column_names(df, year, column_mapping)
    
    # 데이터 구조 변환
    data_list = []
    for year in [2017, 2018, 2019, 2020, 2022, 2023]:
        cols = [col for col in df.columns if col.endswith(f'_{year}')]
        if not cols:
            continue
        temp_df = df[['customer_id'] + cols].copy()
        temp_df.columns = ['customer_id'] + [col.rsplit('_', 1)[0] for col in cols]
        temp_df['year'] = year
        data_list.append(temp_df)
    
    df_long = pd.concat(data_list, ignore_index=True)
    
    # 비연도별 칼럼 병합
    non_yearly_cols = [col for col in df.columns if not any(col.endswith(f'_{year}') for year in [2017, 2018, 2019, 2020, 2022, 2023]) and col != 'customer_id']
    df_non_yearly = df[['customer_id'] + non_yearly_cols].copy()
    df_long = df_long.merge(df_non_yearly, on='customer_id', how='left')
    
    # 연도 컬럼 정렬
    df_long['year'] = df_long['year'].astype(int)
    df_long = df_long.sort_values(['customer_id', 'year'])
    
    # 결측치 처리
    df_long.replace(' ', np.nan, inplace=True)
    
    totalcharges_columns = [col for col in df_long.columns if 'totalcharges' in col]
    for col in totalcharges_columns:
        if col in df_long.columns:
            df_long[col] = pd.to_numeric(df_long[col], errors='coerce')
            mode_value = df_long[col].mode()[0]
            df_long[col].fillna(mode_value, inplace=True)
    
    numeric_columns = df_long.select_dtypes(include=['int64', 'float64']).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in totalcharges_columns and col not in ['customer_id', 'year', 'churn']]
    
    for col in numeric_columns:
        if col in df_long.columns:
            df_long[col] = pd.to_numeric(df_long[col], errors='coerce')
            mean_value = df_long[col].mean()
            df_long[col].fillna(mean_value, inplace=True)
    
    if 'churn' in df_long.columns:
        df_long['churn'] = df_long['churn'].astype(int)
    
    # 범주형 변수 인코딩
    object_columns = df_long.select_dtypes(include=['object']).columns.tolist()
    categorical_columns = list(set(object_columns + [
        'contract', 'paymentmethod', 'internetservice', 'internettype', 'gender', 'churn', 
        'paperlessbilling', 'partner', 'dependents', 'phoneservice', 'multiplelines', 
        'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 
        'streamingmovies'
    ]))
    
    le_dict = {}
    for col in categorical_columns:
        if col in df_long.columns:
            df_long[col] = df_long[col].astype(str)
            df_long[col].fillna('Unknown', inplace=True)
            le = LabelEncoder()
            df_long[col] = le.fit_transform(df_long[col])
            le_dict[col] = le
    
    # 수치형 변수 스케일링
    scaler = StandardScaler()
    numeric_features = df_long.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_features = [col for col in numeric_features if col not in ['customer_id', 'year', 'churn']]
    df_long[numeric_features] = scaler.fit_transform(df_long[numeric_features])
    
    # 시계열 데이터 준비
    grouped = df_long.groupby('customer_id')
    
    sequences = []
    labels = []
    customer_ids = []
    
    for customer_id, group in grouped:
        group = group.sort_values('year')
        features = group.drop(['customer_id', 'year', 'churn'], axis=1).values
        label = group['churn'].iloc[-1]
        sequences.append(features)
        labels.append(label)
        customer_ids.append(customer_id)
    
    max_seq_length = max(len(seq) for seq in sequences)
    n_features = sequences[0].shape[1]
    
    sequences_padded = pad_sequences(sequences, maxlen=max_seq_length, dtype='float32', padding='post', truncating='post')
    labels = np.array(labels)
    customer_ids = np.array(customer_ids)
    ###############################################################################################################################################
    from sklearn.utils import shuffle
    customer_ids_shuffled, sequences_padded_shuffled, labels_shuffled = shuffle(customer_ids, sequences_padded, labels, random_state=42)
    
    split_index = int(len(customer_ids_shuffled) * 0.8)
    X_train = sequences_padded_shuffled[:split_index]
    y_train = labels_shuffled[:split_index]
    X_test = sequences_padded_shuffled[:]
    y_test = labels_shuffled[:]
    ################################################################################################################################################
    overlapping_ids = set(customer_ids_shuffled[:split_index]).intersection(set(customer_ids_shuffled[split_index:]))
    print(f"훈련 세트와 테스트 세트 간 중복된 고객 수: {len(overlapping_ids)}")
    
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print("훈련 데이터 레이블 분포:", dict(zip(unique_train, counts_train)))
    print("테스트 데이터 레이블 분포:", dict(zip(unique_test, counts_test)))
    
   
    
    return X_train, X_test, y_train, y_test, n_features, max_seq_length


@register_keras_serializable()
def focal_loss_fixed(y_true, y_pred, gamma=2., alpha=.25):
    eps = 1e-12
    y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
    pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    loss = -tf.reduce_mean(alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt))
    return loss

# focal_loss 함수는 이제 focal_loss_fixed를 반환하지 않고
# 특정 파라미터로 설정된 함수를 직접 사용할 수 있습니다.

def focal_loss(gamma=2., alpha=.25):
    return lambda y_true, y_pred: focal_loss_fixed(y_true, y_pred, gamma=gamma, alpha=alpha)

# LSTM 모델 빌드 함수
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Masking(mask_value=0.))
    
    # LSTM 레이어 + L2 정규화 적용
    model.add(LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    # 두 번째 LSTM 레이어 + L2 정규화 적용
    model.add(LSTM(64, kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    
    # 출력층
    model.add(Dense(1, activation='sigmoid'))
    
    # 모델 컴파일 (AUC, Precision, Recall, Accuracy 포함)
    model.compile(
        optimizer=Adam(learning_rate=0.0001, clipvalue=1.0),  # 그래디언트 클리핑 적용
        loss=focal_loss(gamma=2., alpha=.25),
        metrics=['AUC', 'Recall']  # 추가된 평가지표
    )
    
    return model

# LSTM 모델 학습 함수
def train_lstm(X_train, y_train, X_test, y_test, input_shape, class_weight_dict):
    # LSTM 모델 빌드
    model = build_lstm_model(input_shape)
    
    # EarlyStopping (validation loss 모니터링)
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, mode='min')
    
    # Learning Rate Scheduler (ReduceLROnPlateau)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
    
    # 모델 학습
    history = model.fit(
        X_train, y_train,
        epochs=40,  # 40 에포크
        batch_size=128,  # 배치 사이즈 128
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        callbacks=[early_stopping, reduce_lr]  # 학습률 스케줄러 및 조기 종료
    )
    
    # 학습 곡선 시각화
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['AUC'], label='Val AUC')
    plt.plot(history.history['val_AUC'], label='Train AUC')
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Val Loss')
    plt.plot(history.history['val_loss'], label='Train Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
    
    # 예측 및 평가
    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    print("LSTM 분류 보고서:")
    print(classification_report(y_test, y_pred))
    print('LSTM ROC-AUC Score:', roc_auc_score(y_test, y_pred_prob))
    
    # 혼동 행렬 시각화
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    
    return model

def save_lstm_model(model, filepath="models/lstm_model.keras"):
    model.save(filepath)
    print(f"LSTM 모델이 {filepath}에 저장되었습니다.")