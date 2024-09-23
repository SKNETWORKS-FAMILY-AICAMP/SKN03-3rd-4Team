import numpy as np

def preprocess_input(input_data):
    """
    입력 데이터를 전처리하는 함수.
    :param input_data: 사용자로부터 입력받은 원본 데이터 (예: 문자열, 리스트)
    :return: 전처리된 numpy array 데이터
    """
    # 예를 들어, 쉼표로 구분된 문자열을 numpy array로 변환
    processed_data = np.array([float(i) for i in input_data.split(',')]).reshape(1, -1)
    
    # 추가적인 전처리 단계 (정규화, 스케일링 등)가 필요하면 여기서 수행
    return processed_data
