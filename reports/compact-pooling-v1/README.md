# 공통 CNN · 집계 방식 3종 · 증강 적용

상태: 자산 복구 후 자동 학습 대기. 결과값은 아직 없다.

- 조건: 전체 1.5초, 검출기/발화확률/외부 사전학습 없음.
- 공통 CNN 구조와 초기값을 맞추고 세 모델을 독립적으로 학습한다. 분류기 초기값도 같다.
- 평균 pooling 77,835 / pooled-query attention 78,859 / self-attention + CLS 86,315 파라미터.
- spk02와 spk03을 각각 홀드아웃하는 두 분할. 매 분할 다른 세 개발 화자의 609개로 학습하고 203개로 검증한다. spk01 음원은 읽지 않는다.
- 각 100에포크, batch32, AdamW 1e-3에서 1e-5까지 cosine, weight decay 1e-4. 한 seed(42).
- 기존 채택 증강: 소음 p=0.5, SNR U[-5,20]dB; 약한 포먼트 p=0.5, loguniform[1/1.1,1.1], 원본별 고정 2변형. 포먼트 후 소음 혼합. 피치·속도·SpecAugment는 사용하지 않는다.
- 매 epoch 모든 모델이 동일한 순서와 동일한 증강 입력을 받는다.
- 기존 평가 소음 8종의 0dB draw0 평균 macroF1로 모델별 체크포인트를 선택한다. 최종 선택/마지막 모델은 원본 및 0dB 3draw로 평가한다. 선택 모델의 잡음 단독 오반응률도 기록한다.
- 개발 2분할의 선별 실험이며 독립 test 결과가 아니다. 기존 Conformer 4분할 결과와는 직접적인 개선 수치로 비교할 수 없다.

실행: `python -m experiments.compact_pooling.run`

실행 로그와 자동 상태: `runs/compact-pooling-v1/{launcher_status.json,progress.json,training.log}`.

실제 음원을 통한 전처리·역전파, 공통 CNN 초기값, 포먼트 변형 및 macroF1 계산을 확인했다. 학습 전에 측정한 GPU 계산 시간은 timing_pilot.json에 기록했다.
