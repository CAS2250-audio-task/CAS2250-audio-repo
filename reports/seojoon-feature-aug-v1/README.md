# 음향 필터링 · 3채널 동적 스펙트로그램(Δ/ΔΔ) · 증강 조합 실험 (`study/seojoon`)

- **작성자**: 이서준 (`study/seojoon` 브랜치)
- **기준 데이터셋**: `datasets/task2/fixed-1p5-v1` (1,015 clips, 16 kHz, 1.5s)
- **평가 원칙**: 독립 테스트 화자 `spk01`은 모든 실험에서 평가하지 않고 미사용 상태로 유지함.

## 1. 실험 동기 및 가설

1. `reference/ai-baseline` (ResNet-18, 10 epochs, `spk05` val) 실행 결과 Accuracy 92.12% / Macro-F1 92.58%였으며, **총 16건의 오분류 중 `후진` 1개 클래스에서만 7건(43.8%)이 집중 발생**(`후진→시작` 3건, `후진→정지` 3건, `후진→재시작` 1건).
2. 원인 분석: `후진`의 초성 `ㅎ`+`/u/` 구간 에너지가 약하고 후속 음절(`-진`)이 `정지/전진/시작`과 유사해 정적 1채널 Log-Mel만으로는 시간축 음절 전이 구분이 어려움. 또한 3채널 ImageNet pretrained ResNet-18을 1채널로 축소해 사용 중이었음.
3. **적용 기법**:
   - **음향 필터링**: `Pre-emphasis (0.97)` + `f_min=80 Hz` (고주파 자음 에너지 강조 및 저주파 대역 정리)
   - **3채널 동적 스펙트로그램**: `[Log-Mel, Δ (1차 시간 미분), ΔΔ (2차 시간 미분)]` 3채널 구성으로 ResNet-18 ImageNet 3채널 Conv1 구조에 직접 매핑
   - **정규화 및 증강**: `use_augmentation=True` (Gain, Circular time shift, Additive noise, SpecAugment Freq/Time masking) + `label_smoothing=0.1`

## 2. 단계별 Ablation 결과 (`spk05` Validation, 203 clips)

| 실험명 | 설정 요약 | Epochs | Best Ep | Val Acc | Macro-F1 | 총 오분류 | `후진` 오분류 |
|---|---|---:|---:|---:|---:|---:|---:|
| `baseline-v1` | 기본 1ch Log-Mel | 10 | 10 | 92.12% | 92.58% | 16 | **7** |
| `exp1_aug` | Augmentation ON | 10 | 9 | 92.61% | 93.07% | 15 | **0** |
| `exp2_filter` | Pre-emphasis 0.97 + f_min=80 | 10 | 9 | 92.61% | 93.16% | 15 | **0** |
| `exp3_delta3ch` | 3ch (Log-Mel + Δ + ΔΔ) 단독 | 10 | 9 | 89.66% | 90.48% | 21 | 1 |
| **`exp4_combo`** | **필터 + 3ch Δ/ΔΔ + Aug + Label Smoothing(0.1)** | **20** | **18** | **97.54%** | **97.72%** | **5** | **0** |

- `exp4_combo`의 남은 오분류 5건: `복귀→대기`(1), `오른쪽→왼쪽`(2), `재시작→시작`(1), `정지→재시작`(1). (`후진` 오분류 7건 → **0건 완전 해소**)

## 3. 동일 20 Epoch 4-Fold 화자 교차검증 (`spk01` Test 제외)

| Validation 화자 | Baseline (20ep) Acc | `exp4_combo` Acc | Baseline (20ep) F1 | `exp4_combo` F1 | F1 변화 |
|---|---:|---:|---:|---:|---:|
| `spk02` | 67.49% | **83.74%** | 70.39% | **84.69%** | **+14.30%p** |
| `spk03` | 84.73% | **89.16%** | 84.74% | **87.47%** | +2.73%p |
| `spk04` | 84.73% | **91.63%** | 85.10% | **90.70%** | +5.60%p |
| `spk05` | 94.09% | **97.54%** | 94.59% | **97.72%** | +3.13%p |
| **4-Fold 평균** | **82.76%** | **90.52%** | **83.70%** | **90.15%** | **+6.45%p** |

### 핵심 인사이트
- 정적 Log-Mel 스펙트로그램은 화자의 기본주파수(F0)에 따라 주파수 축 패턴이 크게 이동하나, **시간 미분(Δ, ΔΔ)과 Pre-emphasis 기반 자음 전이 특징은 화자 음높이 변화에 상대적으로 불변(speaker-invariant)**하여 미학습 화자(`spk02` 등) 일반화 성능을 크게 개선함.

## 4. 혼동 행렬 비교

- `spk05` Baseline vs `exp4_combo`: `assets/baseline_spk05_cm.png`, `assets/exp4_combo_spk05_cm.png`
- `spk02` Baseline(20ep) vs `exp4_combo`: `assets/baseline_e20_spk02_cm.png`, `assets/exp4_combo_spk02_cm.png`

## 5. 재현 명령어

- 학습: `python3 -m src train --config configs/seojoon/exp4_combo.json --output runs/seojoon-exp4_combo --device cuda`
- 평가: `python3 -m src evaluate --run runs/seojoon-exp4_combo --split val`

## 6. 백본 경량화 · 추론 Latency · 0dB 백색소음 스트레스 벤치마크 (`spk05` Val)

실시간 데모(`command-rescue`) 및 경량 배포 적합성을 판단하기 위해, 1.5초 단일 클립 추론 시간(100회 중앙값)과 결정론적 0dB 백색소음 환경 성능을 비교함.

| 모델 | 백본 | Params | Clean Acc | Clean F1 | 0dB Noise Acc | 0dB Noise F1 | CPU (ms) | GPU (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline-v1` | ResNet-18 (1ch) | 11.18M | 92.12% | 92.58% | 1.48% | 0.26% | 93.23 | 1.98 |
| **`exp4_combo`** | **ResNet-18 (3ch Δ)** | **11.18M** | **97.54%** | **97.72%** | **67.98%** | **68.09%** | 93.39 | **2.02** |
| `exp5_mobilenet` (`lr=5e-4, 20ep`) | MobileNetV3-Small | 1.53M | 78.82% | 79.02% | 30.05% | 25.85% | **10.63** | 4.04 |
| `exp5_mobilenet_tuned` (`lr=2e-3, 30ep`) | MobileNetV3-Small | **1.53M** | 92.12% | 92.76% | 43.35% | 41.92% | 51.26 | 4.07 |

### 결론
- **실시간 GPU 데모/서버 추론**: `exp4_combo`가 **2.02 ms**의 초저지연으로 Clean **97.54%**, 0dB 소음 **67.98%**를 달성해 가장 균형이 뛰어남 (기본 ResNet-18은 0dB 소음에서 1.48%로 붕괴).
- **CPU·모바일 경량 배포**: `MobileNetV3-Small`은 파라미터를 **7.3배 절감(1.53M)**하면서도 학습률 튜닝(`lr=2e-3, 30ep`) 시 Clean F1 **92.76%**를 유지함.
