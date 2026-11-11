# Validation 오류 사례 점검

seed42 기준과, 반복 검증에서 가장 높았던 보조 세트 4를 비교한다. 전체 8세트 중 오류가 나온 원본 validation 파일에서 정답 단어별로 순환 선택한 최대 20개다. 아래는 예측과 추출 상태로 확인한 관찰이며, 청취에 근거한 오류 원인 판정은 하지 않았다.

F1 측정 가능 비율은 정확도나 발화 비율이 아니다. Softmax 값도 보정된 정답 확률이 아니다.

| 파일 | 정답 | 기준 예측 | 세트 4 예측 | 관찰 | F1 측정 가능 프레임 |
| --- | --- | --- | --- | --- | --- |
| spk05_004 | 대기 | 대기 | 재시작 | introduced_error | 59.6% |
| spk05_042 | 복귀 | 대기 | 대기 | both_incorrect | 48.3% |
| spk05_063 | 시작 | 시작 | 재시작 | introduced_error | 69.5% |
| spk05_081 | 오른쪽 | 왼쪽 | 왼쪽 | both_incorrect | 68.2% |
| spk05_088 | 왼쪽 | 왼쪽 | 왼쪽 | both_correct | 64.2% |
| spk05_112 | 재시작 | 시작 | 시작 | both_incorrect | 50.3% |
| spk05_130 | 전진 | 대기 | 대기 | both_incorrect | 56.3% |
| spk05_162 | 정지 | 정지 | 대기 | introduced_error | 69.5% |
| spk05_180 | 출발 | 시작 | 후진 | both_incorrect | 55.0% |
| spk05_189 | 후진 | 시작 | 대기 | both_incorrect | 53.0% |
| spk05_032 | 복귀 | 시작 | 정지 | both_incorrect | 62.3% |
| spk05_059 | 시작 | 재시작 | 재시작 | both_incorrect | 65.6% |
| spk05_066 | 오른쪽 | 왼쪽 | 오른쪽 | fixed_baseline_error | 74.8% |
| spk05_107 | 재시작 | 시작 | 재시작 | fixed_baseline_error | 55.0% |
| spk05_138 | 전진 | 전진 | 전진 | both_correct | 65.6% |
| spk05_154 | 정지 | 정지 | 정지 | both_correct | 53.0% |
| spk05_181 | 출발 | 출발 | 전진 | introduced_error | 57.6% |
| spk05_191 | 후진 | 시작 | 후진 | fixed_baseline_error | 53.6% |
| spk05_033 | 복귀 | 정지 | 복귀 | fixed_baseline_error | 65.6% |
| spk05_055 | 시작 | 시작 | 왼쪽 | introduced_error | 62.9% |
