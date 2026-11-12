# 화자별 개발 CV 실험

현재 단계: **complete**, 완료 학습: **184회**.

데이터는 real-context-1p5-v1, 개발 화자는 spk02~05입니다. 각 fold train 609개 / validation 203개이며 silence도 연결된 화자와 함께 이동합니다. 기존 184회 CV에서는 spk01을 평가하지 않았습니다. 선택 완료 후 사용자가 요청한 별도 진단은 아래에 구분했습니다.

주 지표는 증강하지 않은 validation의 화자 4-fold 평균 macro F1입니다. fold마다 ImageNet 초기값에서 새 모델을 학습하고 scaler를 train에서만 적합합니다. 개발 CV로 조건과 epoch를 선택하므로 독립 최종 평가 점수로 해석하지 않습니다.

all_runs.csv는 fold별 결과, seed_summary.csv는 세트·seed별 CV와 최저 화자 점수입니다. condition_summary.csv에는 seed 수가 다른 선별/확인 조건이 섞여 있으므로 seeds 열을 함께 봅니다.

파형 증강은 클립당 고정 2개 변형을 준비하고, 매 학습 draw에서 각 정책을 확률 0.5로 독립 적용합니다. 검증에는 원본만 사용합니다. 이 첫 실험은 무한히 새로운 파형을 생성하는 온라인 증강의 성능을 평가한 것은 아닙니다.

6개 피크 예외 파일의 원본 검수 구간 보존과 신호 수치는 확인했으며 사람의 청취 검수는 아직 수행하지 않았습니다. 자동 검증 결과를 청취 완료로 표현하지 않습니다.

실행별 best checkpoint와 재개 checkpoint는 runs/auxiliary-cv-v1에 보관합니다. 중단 후 같은 pipeline 명령을 실행하면 완료 fold를 건너뛰고 미완료 epoch부터 재개합니다.

**완료 결과**

정식 학습 184회(1840 epochs)를 완료했다. 별도의 복구 검증 4 epochs는 조건 선택에 쓰지 않았다.

초기 무증강 3시드 CV F1: set0 76.76%, set4 80.32%.
선택한 공통 조건의 3시드 CV F1: set0 82.51%, set4 83.42%.
공통 조건: `{"augmentation": ["mask", "speed"], "backbone_lr_ratio": 1.0, "dropout": 0.0, "freeze_front": false, "weight_decay": 0.0001}`. 최종 선택 세트: 3.

전체 세트의 공정한 한 시드 비교는 feature_sets_cv.png, 초기/선택 조건의 화자별 3시드 비교는 speaker_stability.png에 있다. 체크포인트 선택·예측·초기값·분할·표준화·프로토콜 해시를 독립적으로 다시 확인했고 run_audit.csv와 execution_audit.json에 기록했다.

전체 세트 표·선택 근거·오답 변화와 한계는 [결과 해석](interpretation.md)에 정리했다.

단계별 학습–검증 격차와 후반 과적합의 실제 근거는 [과적합 분석](overfitting.md)에 정리했다.

후속 사용자 요청으로 [화자별 악화 파일 진단](file_diagnostics/README.md)을 추가했다. spk02·03·04의 seed44 학습을 동일하게 재현해 epoch별 예측을 복원했고, spk01은 선택 완료 후 별도로 평가했다. 이 후속 분석은 기존 184회 CV·최종 조건 선택과 구별한다.

이어서 같은 조건의 세트0을 재현해 [보조 제외/포함 파일별 비교](file_diagnostics_comparison/README.md)를 완료했다. 각 화자 203개를 짝지어 epoch3→10 예측·악화/회복·공통 초반 정답 집단·단어별 정답 수를 비교했다. 보조 없는 원자료는 [세트0 진단](file_diagnostics_set0/README.md)에 있다.

[각 모델의 최고 성능 체크포인트끼리 비교](best_checkpoint_comparison/README.md)도 추가했다. 원래 검증 F1로 선택한 epoch를 사용하며 화자2는 세트0 epoch4 62.82%, 세트3 epoch3 70.14%다. 화자1은 화자2에서 선택한 체크포인트를 그대로 평가했다. 각 화자 문서에 한쪽만 정답·둘 다 오답인 모든 파일과 WAV 링크가 있다.

화자별 최고 epoch 선택을 없앤 [총합 기준 공통 최고 epoch 비교](common_epoch_comparison/README.md)도 완료했다. 3시드×4-fold 전체 평균 F1이 최고인 공통 epoch를 세트마다 하나씩 골랐다. 세트0은 epoch8 79.99%, 세트3은 epoch9 82.30%(+2.32%p)다. 같은 epoch를 모든 화자·시드에 적용한 결과이며, 별도 spk01은 선택에서 제외했다. seed44의812개 파일 짝 비교·화자5 포함 목록·별도 화자1 진단과 전체 학습곡선도 제공한다.
