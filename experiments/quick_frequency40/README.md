# 40분 이내 구조 가설 예비 비교

6조건, Pooled 하나, 화자03/seed42/동일 증강/CE+0.1SupCon.
모두 새로300epoch 학습하되1000epoch cosine 일정의 첫300epoch를 사용한다.
기존1000epoch 최고점과 직접 비교하지 않고 새 대조군끼리 비교한다.

- low8: 기존 소형2D 전처리 대조군.
- low16: 전처리만8→16채널.
- low_axis8:8채널 유지,3×3 두 층을5×1 및1×3으로 분리. 시간 support도 달라지므로 순수 주파수효과라고 단정하지 않는다.
- freq_avg: 기존 세로CNN 전체 주파수 평균 대조군.
- freq_band2: 최종 및 중간 경로의 주파수를2구역 평균/연결 후 투영. 동일 출력 초기화.
- freq_capacity: 전체 주파수 평균 유지 +32→86→32 residual MLP. 추가5,622파라미터로band2의 추가5,632개와 근접하게 맞춘 용량 대조군. 동일 출력 초기화.

세로 어텐션은 전부OFF. CNN블록 및local 중간 전달은 유지한다.
기존 trainer/model 코드를 수정하지 않고 trainer 사본에 schedule-epochs 옵션만 추가했다.
모델별 분류기와SupCon projector는 기존3head 실행의Pooled 초기값으로 고정한다.

진행: runs/quick-frequency40/status.json. 준비 시점부터36분 제한이며 진행 중 자식까지 자동 종료한다.
완료 조건마다 reports/quick-frequency40/comparison.csv에 원본·0dB·명령어·silence·잡음오반응을 저장한다.
시간 제한으로 중단하면 마지막10epoch 단위 optimizer checkpoint와 best를 보존한다.
300epoch 예비 결과는1000epoch 최종 성능이나 검출기 적용 후 순위를 보장하지 않는다.
