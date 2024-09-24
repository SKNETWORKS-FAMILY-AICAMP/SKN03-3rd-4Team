from django.shortcuts import render

# Create your views here.

# - [ ]  이탈 확률이 높은 고객에게 적절한 솔루션 제안 (맞춤형 솔루션 제공)
# - [ ]  이탈 고객군 분류 및 분석
# - [ ]  고객 특성에 따른 맞춤형 솔루션 제시 (장기 고객 혜택, 단기 고객 유도 전략 등)


# - [ ]  이탈 고객 유지 방안 설계 (할인 제공, 맞춤형 서비스 제공 등)
# - [ ]  고객별 맞춤형 솔루션을 제공하는 기능 구현

# request를 받음 
#  
# prediction 페이지에서 선택된 요소를 우리 페이지로 가져오는 로직이 필요
# prediction 요소를 리스트화? (받아올 요소는 몇 가지인가)

# solution 답변을 모두 리스트화(딕셔너리?)
#

# 만약 prediction 에서 a일 경우 => a1 답변을 출력 ex) 남자일 경우 ~~~한 답변을 출력
# 만약 prediction 에서 b일 경우 => b1 답변을 출력.....
# 해당 로직을 간소화 할 모듈이 필요

# 그런데 만약 prediction에서 a와 b가 겹칠 때 (솔루션이 있다면) => a and b 답변을 출격
# 

# 솔루션 매핑 (각 조건에 따른 답변을 리스트로 구성)

# 솔루션 매핑 (각 조건에 따른 답변을 리스트로 구성)
solutions = {
    'male': '남성 고객에게는 특별한 남성용 프로모션을 제공하세요.',
    'female': '여성 고객에게는 여성 전용 혜택을 추천합니다.',
    'long_term': '장기 고객에게는 추가 할인을 제공하세요.',
    'short_term': '단기 고객을 위한 유도 전략을 마련하세요.',
    'high_spender': '높은 소비 고객에게 VIP 서비스를 추천합니다.',
    'low_spender': '소비가 적은 고객에게는 맞춤형 소액 상품을 제안하세요.'
}

def get_customer_solution(predictions):
    """
    predictions: 예측 데이터 (리스트 형태로 여러 요소가 들어옴)
    각 prediction에 대한 적절한 솔루션을 리스트 형태로 반환
    """
    solution_list = []

    # 1. 가입 기간이 a일 경우
    if int(predictions.get('tenure', 0)) < 3:
        solution_list.append("가입 기간이 짧습니다. 장기 유지 프로모션을 제공하세요.")

    # 2. 계약 기간이 a일 경우
    if predictions.get('contract') and int(predictions.get('contract')) < 1:
        solution_list.append("계약 기간이 짧습니다. 장기 계약 할인 제공을 고려하세요.")

    # 3. 월 비용이 a 이상일 경우
    if predictions.get('monthly_charges') and float(predictions.get('monthly_charges')) > 50:
        solution_list.append("월 비용이 높습니다. 월 요금 할인을 제공하거나 저렴한 요금제를 추천하세요.")

    # 4. 총 비용이 a 이상일 경우
    if predictions.get('totalCharges') and float(predictions.get('totalCharges')) > 600:
        solution_list.append("총 비용이 높습니다. VIP 혜택을 추천하거나 추가 할인을 제공하세요.")

    # 5. 결제 방식이 특정 방식일 경우
    if predictions.get('payment_method') == '1':  # '전자 수표'일 경우
        solution_list.append("결제 방식이 불편할 수 있습니다. 자동 은행 이체 또는 신용카드 결제를 추천하세요.")

    # 6. 폰 서비스가 없는 경우
    if predictions.get('phoneService') == '2':  # '아니오'일 경우
        solution_list.append("전화 서비스가 없습니다. 번들 서비스를 제공하여 전화 서비스를 추가하세요.")

    # 7. 종이 영수증을 받는 경우
    if predictions.get('paperlessBilling') == '2':  # '아니오'일 경우
        solution_list.append("종이 청구서를 받고 있습니다. 전자 청구서로 전환하도록 유도하세요.")

    # 8. 다중 회선을 사용하지 않는 경우
    if predictions.get('multipleLines') == '2':  # '아니오'일 경우
        solution_list.append("다중 회선을 사용하지 않습니다. 다중 회선 사용을 장려하세요.")

    # 9. 인터넷 보안이 되어 있지 않을 경우
    if predictions.get('onlineSecurity') == '2':  # '아니오'일 경우
        solution_list.append("인터넷 보안 서비스가 없습니다. 보안 서비스를 추가하여 보호 혜택을 제공하세요.")

    # 10. 온라인 백업이 되어 있지 않을 경우
    if predictions.get('onlineBackup') == '2':  # '아니오'일 경우
        solution_list.append("온라인 백업 서비스가 없습니다. 온라인 백업 서비스를 추가하도록 권장하세요.")

    # 11. 장치 보호가 되어 있지 않을 경우
    if predictions.get('deviceProtection') == '2':  # '아니오'일 경우
        solution_list.append("장치 보호 서비스가 없습니다. 장치 보호 서비스를 추가하세요.")

    # 12. 기술 지원을 받지 않는 경우
    if predictions.get('techSupport') == '2':  # '아니오'일 경우
        solution_list.append("기술 지원 서비스가 없습니다. 기술 지원 서비스를 추천하세요.")

    # 13. TV 스트리밍을 받지 않는 경우
    if predictions.get('streamingTV') == '2':  # '아니오'일 경우
        solution_list.append("TV 스트리밍 서비스를 이용하지 않습니다. TV 스트리밍 서비스를 추가하세요.")

    # 14. 영화 스트리밍을 받지 않는 경우
    if predictions.get('streamingMovies') == '2':  # '아니오'일 경우
        solution_list.append("영화 스트리밍 서비스를 이용하지 않습니다. 영화 스트리밍 서비스를 추천하세요.")

    return solution_list


def prediction_solution_view(request):
    if request.method == 'POST':
        # 폼에서 전송된 데이터 가져오기
        predictions = {
            'gender': request.POST.get('gender'),
            'partner': request.POST.get('partner'),
            'tenure': request.POST.get('tenure'),
            'contract': request.POST.get('contract'),
            'monthlyCharges': request.POST.get('monthlyCharges'),
            'paymentMethod': request.POST.get('paymentMethod'),
            'dependents': request.POST.get('dependents'),
            'phoneService': request.POST.get('phoneService'),
            'paperlessBilling': request.POST.get('paperlessBilling'),
            'multipleLines': request.POST.get('multipleLines'),
            'onlineSecurity': request.POST.get('onlineSecurity'),
            'onlineBackup': request.POST.get('onlineBackup'),
            'deviceProtection': request.POST.get('deviceProtection'),
            'techSupport': request.POST.get('techSupport'),
            'streamingTV': request.POST.get('streamingTV'),
            'streamingMovies': request.POST.get('streamingMovies'),
            'totalCharges': request.POST.get('totalCharges')
        }

        # 고객 맞춤형 솔루션 제공
        customer_solutions = get_customer_solution(predictions)

        # context에 데이터 추가
        context = {
            'gender': predictions.get('gender'),
            'partner': predictions.get('partner'),
            'tenure': predictions.get('tenure'),
            'contract': predictions.get('contract'),
            'monthlyCharges': predictions.get('monthlyCharges'),
            'paymentMethod': predictions.get('paymentMethod'),
            'dependents': predictions.get('dependents'),
            'phoneService': predictions.get('phoneService'),
            'paperlessBilling': predictions.get('paperlessBilling'),
            'multipleLines': predictions.get('multipleLines'),
            'onlineSecurity': predictions.get('onlineSecurity'),
            'onlineBackup': predictions.get('onlineBackup'),
            'deviceProtection': predictions.get('deviceProtection'),
            'techSupport': predictions.get('techSupport'),
            'streamingTV': predictions.get('streamingTV'),
            'streamingMovies': predictions.get('streamingMovies'),
            'totalCharges': predictions.get('totalCharges'),
            'solutions': customer_solutions
        }

        return render(request, 'solution.html', context)

    return render(request, 'prediction.html')

