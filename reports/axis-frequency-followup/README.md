# 실험 C 후속: 축 분리 전처리 + 두 구역 주파수 CNN

A와 B 모두 실제 500 epoch 학습과 선택 체크포인트의 최종 평가를 완료했다. 기존 500 epoch 두 구역 CNN과도 조건을 대조했다.

- A: 기본 Mel → 8채널 축 분리 잔차 전처리(5×1 → 1×3) → 두 구역 주파수 CNN → 시간 Pooled → 11클래스.
- B: A의 전처리 직후, CNN 직전에 기존 FrequencyAttention 1블록을 추가.
- Frontend와 BandRoutes는 quick_frequency40 구현을 재사용한다. CNN 본체는 기존 FrequencyTimeCNN의 주파수·시간 분리 블록이며, 중간 48채널과 최종 128채널 각각의 저·고 대역을 평균/연결해 32차원으로 투영하고 더한다.
- 전처리와 세로 어텐션의 잔차 출력은 0으로 초기화했다. CNN·Pooled·분류기·SupCon projector 초기값을 기존 대조군과 맞췄다.

D1 fixed-1p5-v1, 16 kHz/1.5초, spk02·04·05 학습 609개 / spk03 검증 203개, seed 42. 기본 64 log10 Mel(20–7800 Hz, FFT512/win400/hop160, 151프레임), 원본 학습 오디오만으로 대역별 정규화를 적합했다.

손실은 paired CE + 0.1 SupCon, 온도 0.1, 같은 명령어를 positive로 사용하고 silence는 SupCon에서 제외했다. 포먼트 증강 p=0.5 뒤 학습 소음 6종 중 하나를 p=1.0, SNR U[-5,20] dB로 혼합했다. 속도·피치·SpecAugment는 사용하지 않았다.

AdamW(lr=0.001, weight decay=0.0001), 원본 기준 배치 32(참조/잡음 쌍 64개), gradient clip=5. 1000 epoch cosine(lr 최저 1e-5)의 첫 500 epoch를 실행했다. 500 epoch에 cosine을 끝까지 내리는 일정이 아니다. 매 epoch 소음 8종의 고정 draw0 0 dB 전체 11클래스 macro F1 평균이 엄격하게 증가할 때 체크포인트를 저장하여 동률이면 가장 이른 epoch를 선택했다.

| 모델 | 선택 epoch | 원본 macro F1 | 0 dB 전체 macro F1 | 잡음 단독 오반응률 | 추론 파라미터 |
|---|---:|---:|---:|---:|---:|
| 기존 두 구역 CNN | 142 | 88.57% | 81.45% | 14.37% (69/480) | 87,947 |
| A | 254 | 90.80% | 81.59% | 17.08% (82/480) | 88,115 |
| B | 276 | 94.47% | 85.29% | 6.88% (33/480) | 90,340 |

0 dB 지표는 소음 8종 × 3구간 각각의 11클래스 macro F1을 구한 뒤 24개를 평균했다. 잡음 단독 평가는 같은 소음 8종 × 3구간 × 20개 = 480개에서 silence 이외를 출력한 비율이다. 학습 전용 SupCon projector 4,192개 파라미터는 추론 파라미터 수에서 제외했다.

| 차이 (%p) | 원본 macro F1 | 0 dB 전체 macro F1 | 잡음 단독 오반응률 | 파라미터 차이 |
|---|---:|---:|---:|---:|
| B − A | +3.67 | +3.70 | -10.21 | +2,225 |
| A − 기존 두 구역 CNN | +2.23 | +0.14 | +2.71 | +168 |

F1은 높을수록, 잡음 오반응률은 낮을수록 좋다. 한 검증 화자·한 seed 결과이며, 선택에 사용한 draw0가 최종 3구간 평가에도 포함된다. 독립 test 결과나 여러 seed에 대한 우열을 뜻하지 않는다.

샘플 배치 forward/backward, 모든 gradient의 유한성, 잔차 전처리·중간 경로·세로 어텐션의 학습 신호를 확인했다. 저장 예측 5,075개/모델에서 F1을 다시 계산했고, 각 500개 학습 이력·선택 기준·학습률·체크포인트 해시·정규화·소음/오디오 해시를 검증했다. A/B/기존 대조군의 500 epoch 증강 trace는 모두 동일하다. 자세한 근거는 preflight.json, completion_audit.json, augmentation_trace_hashes.csv에 보존했다.

## 실행 명령

```bash
bash experiments/run_axis_frequency_followup.sh
```

개별 학습 명령:

```bash
.venv/bin/python -u -m experiments.low_frequency_grid.run --architecture axis_frequency --frontend standard --folds spk03 --objective paired_supcon --positive-mode same_command --supcon-weight 0.1 --temperature 0.1 --epochs 1000 --schedule-epochs 1000 --stop-after 500 --output runs/lowfreq-axis_frequency-standard-spk03-e1000
.venv/bin/python -u -m experiments.low_frequency_grid.run --architecture axis_frequency_attn --frontend standard --folds spk03 --objective paired_supcon --positive-mode same_command --supcon-weight 0.1 --temperature 0.1 --epochs 1000 --schedule-epochs 1000 --stop-after 500 --output runs/lowfreq-axis_frequency_attn-standard-spk03-e1000
```

두 구성은 기존 experiments.low_frequency_grid.run 진입점의 --architecture 선택지로 사용할 수 있다. 프런트엔드/평가/손실/체크포인트 선택은 기존 trainer를 그대로 사용한다. 재실행 시 일치하는 protocol의 마지막 optimizer 체크포인트에서 이어서 실행한다. 의존성은 저장소 .venv에 soundfile==0.13.1, praat-parselmouth==0.4.6을 설치했고, 기존 torch/torchaudio 2.3.1+cu121과 RTX 3090을 사용했다.

## 저장 경로

- 결과표: `reports/axis-frequency-followup/comparison.csv` 및 `comparison.json`.
- 실행 명령: `reports/axis-frequency-followup/commands.json`.
- 학습 로그: `runs/axis-frequency-followup/axis_frequency.log`, `axis_frequency_attn.log`.
- A 선택 체크포인트: `runs/lowfreq-axis_frequency-standard-spk03-e1000/spk03/cnn_pooled_query_best.pt`.
- A 마지막 epoch500 + optimizer: `runs/lowfreq-axis_frequency-standard-spk03-e1000/spk03/last.pt`.
- A 전체 이력·증강 trace·최종 예측/지표: `runs/lowfreq-axis_frequency-standard-spk03-e1000/spk03/`; 고정 protocol: `reports/lowfreq-axis_frequency-standard-spk03-e1000/protocol.json`.
- B 선택 체크포인트: `runs/lowfreq-axis_frequency_attn-standard-spk03-e1000/spk03/cnn_pooled_query_best.pt`.
- B 마지막 epoch500 + optimizer: `runs/lowfreq-axis_frequency_attn-standard-spk03-e1000/spk03/last.pt`.
- B 전체 이력·증강 trace·최종 예측/지표: `runs/lowfreq-axis_frequency_attn-standard-spk03-e1000/spk03/`; 고정 protocol: `reports/lowfreq-axis_frequency_attn-standard-spk03-e1000/protocol.json`.
- 기존 비교 기록: `reports/lowfreq-frequency-standard-spk03-e1000/summary.json`.
