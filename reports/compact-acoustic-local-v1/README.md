# local + 보조 음향 set3 (완료)

기준 origin/compact-pooling-v1 @b00956b에서 분기한 experiment/compact-local-acoustic-aux. 병합된 local 구조의 Mean/Pooled/Self에 예전 F1/F2/F3/B1/B2/B3 보조 입력을 추가한다. 주파수축 ablation은 이 실험에서 켜지 않는다.

- 화자03 검증, 나머지 개발화자02/04/05 학습. seed42, 1000epoch, CE+0.1 SupCon same_command, 기존 LR/batch/증강 유지.
- 추가입력은 포먼트/대역폭6개와 측정가능6개. 실제 원본·잡음혼합파형에서 각각 추출하며 train 원본609개 유효프레임으로만 scaler를 적합한다.
- 추가 CNN12→16→32→32, kernel5/5/1, 시간151프레임 유지. 최종32D 특징에 더하고 마지막 projection은0초기화. CNN/readout 세 모델은 독립 학습한다.
- 파라미터 Mean84,635 / Pooled85,659 / Self93,115. 각4,720개 증가. scaler는 checkpoint buffer로 보존한다.
- 학습원본과 augmentation은 그대로이며 별도 손실항을 추가한 실험이 아니라 보조 입력 경로 추가다. 추론에도 음향 추출/보조 경로가 필요하다.
- 공통 CNN/로스/주파수실험 코드는 수정하지 않았다. 기존 local 완료 결과를 대조군으로 재사용한다.
- 전체macro F1 외에 정답률, 명령어10종 F1, silence F1, 정지↔전진 및 후진 오답을 함께 검토한다. 현재 소음 평가 최고점 선택, 동일 개발검증 재평가이므로 독립test가 아니다.

상태: runs/compact-acoustic-local-v1/status.json
로그: runs/compact-acoustic-local-v1/training.log
에포크/ETA: runs/compact-acoustic-local-spk03-e1000/progress.json
완료 후 comparison.csv와epoch_histories.csv.gz,completion_audit.json을 자동 생성한다.

[구현/재현](../../experiments/acoustic_local/README.md) · [사전검사](preflight.json)

[병합 전달: 평균/어텐션 비교, 주파수축 기대와 보조경로의 관계](../../docs/COMPACT_MODEL_DESIGN_HANDOFF.md)

**완료:** 세 모델 각1000epoch 정상 종료. [결과 검토](RESULTS.md)를 확인한다. Mean은0dB F1이개선됐으나잡음단독오반응이증가했고Pooled/Self는0dB F1이하락했다.
