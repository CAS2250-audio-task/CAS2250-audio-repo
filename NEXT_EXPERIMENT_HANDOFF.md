# 음성 명령 인식 실험 — 다음 채팅 인수인계

2026-11-11 작성, 한국 시간 기준. 이 파일은 앞 대화를 읽지 않은 다음 작업자가 실험을 이어가기 위한 요약이다.

**현재 완료:** 보조 음향 특징 0~7세트의 첫 실험 총 16회, 결과 기록·개인 브랜치 업로드, 추가 추론 진단.
**다음 단계:** 화자별 4-fold CV에서 증강·과적합 완화 조건을 비교한 뒤 보조 특징 효과를 재검증.
**CV와 학습 증강은 아직 구현·실행하지 않았다. 현재 실행 중인 학습도 없다.**
사용자의 이번 요청은 이 인수인계 파일 작성이다. 아래 세부 범위와 수치는 다음 실험의 제안값이며 실행 결과가 아니다.

**사용자 의도와 작업 원칙**

- 사용자는 실험을 요청하면 코드 작성 → 실행 → 검증 → 기록까지 진행하기를 원한다. 설명은 한국어로 간결하게 한다.
- 사용자 개인 영역은 `jaehun`이다. 공통 ref를 임의로 변경하지 않는다.
- 원본 음성은 반드시 보존한다. 검수한 1.5초 데이터와 기존 실험 결과도 덮어쓰지 않는다.
- 사용자는 피치 증강, 보조 음향 특징, 향후 멀티태스크/다른 head 실험에 관심이 있다. 뒤쪽 512차원 표현을 재사용할 수 있게 유지한다.
- 기존 역할 논의에서 팀원은 증강·파라미터·전처리·필터링을 맡기로 했다. 최근에는 사용자가 직접 CV·증강·과적합 완화 비교에도 관심을 보였다. 팀원의 변경사항이 새로 있다면 먼저 확인한다.
- 새 실험은 `experiments/`에 별도로 구현하는 방식이 적절하다. 기존 `src/`는 reference 구현이다.

**저장소와 실행 환경**

- 작업 경로: `/root/audio`
- GitHub: https://github.com/CAS2250-audio-task/CAS2250-audio-repo
- 로컬 브랜치 `jaehun` → 원격 `origin/study/jaehun`.
- 인수인계 작성 직전 HEAD/원격 개인 브랜치: `ed58e343f393ce9e0e59769f6c6119c86f739852`.
- 공통 ref: `origin/reference/ai-baseline`, 커밋 `6d208bc0eead31d179e63d050a26ff4511a8fee9`.
- 관련 개인 커밋: `8940edd` 실험 구현, `3bb3f4d` 디지털 무음 분석 창 처리, `ed58e34` 결과 기록.
- Python: `/root/audio/.venv/bin/python`, Python 3.10.12. GPU RTX 3090 24 GB.
- 현재 torch/torchvision/torchaudio: `2.6.0+cu124 / 0.21.0+cu124 / 2.6.0+cu124`.
- numpy 1.24.4, pandas 2.2.2, scipy 1.14.0, sklearn 1.5.1, librosa 0.11.0, praat-parselmouth 0.4.6.
- `.venv`는 system-site-packages를 포함한다. 일부 버전이 루트 `requirements.txt`와 다르므로 무조건 재설치하지 않는다.
- 수치 스택 재현용: `experiments/auxiliary/requirements-reproduce.txt`.
- 실제 환경 기록: `reports/auxiliary-v1/environment.json`.
- 명령은 저장소 루트에서 실행한다. 보통 `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1`을 사용했다.
- 이 세션의 일반 shell sandbox는 `bwrap ... Permission denied`로 실패해서 작업 범위에 맞는 `require_escalated` 실행을 사용했다. 다음 환경의 권한 정책을 따른다.
- 코드·보고서는 Git에 있지만 루트 `runs/`, raw WAV/M4A와 원본 ZIP 일부는 Git에서 제외된다. 다른 머신에서 이어갈 경우 실제 파일 존재부터 확인한다.

**데이터 상태**

- 현재 학습 데이터: `datasets/task2/fixed-1p5-v1/`.
- WAV 1,015개 = 10단어 × 100개 + silence 15개.
- 모두 mono PCM16, 16 kHz, 정확히 24,000 samples = 1.5초.
- 라벨 순서: `출발, 정지, 전진, 후진, 왼쪽, 오른쪽, 시작, 복귀, 대기, 재시작, silence`.
- 한 화자당 단어 200개 + 연결된 silence 3개 = 203개.
- 기존 분할: train `spk02/03/04` 609개, val `spk05` 203개, test `spk01` 203개.
- Silence는 `speaker_id`가 비어 있을 수 있다. 현재 `group`은 `speaker:spkXX`, `room_session_id`도 기록되어 있으므로 이를 확인해 화자 묶음과 함께 이동한다.
- 원본 보관: `datasets/task2/raw-v1/audio/`의 WAV 1,015개, `datasets/task2/raw-v1/originals/`의 M4A 1,015개, `documents/audio_real_dataset.zip`.
- 기존 클리핑은 사용자 청취 검토를 거쳤다. 긴 8개는 선택 문자열 `11111611`을 적용했고 나머지는 기존 recipe에 기록되어 있다.
- 고정 데이터는 VAD 구간 기반 클리핑/패딩과 제한된 음량 조절을 이미 거친 상태다. 기존 silence는 증폭하지 않았다. 원본을 시간 늘이기로 강제 1.5초화한 데이터가 아니다.
- 클리핑 코드는 `preprocessing/clip_audio.py`에 분리되어 있다. 새 실험을 위해 WAV 자체를 덮어쓰지 않는다.
- 기존 1,015개 출력 해시, raw 개수와 원본 ZIP 해시는 최종 감사에서 확인했다: `reports/auxiliary-v1/execution_audit.json`.

**고정 baseline과 보조 특징 모델**

- 학습 기본 설정은 사용자가 baseline으로 지정한 70% 노트북 설정을 따른다.
- `configs/task2_real.json`: epochs 10, batch 32, LR `5e-4`, AdamW weight decay `1e-4`, cosine, inverse-frequency WeightedRandomSampler, ImageNet pretrained ON, augmentation OFF.
- log-mel: n_fft/win 512, hop 160, mel 128, 40~8000 Hz, power→dB top_db 80→기존 clamp/scale. 입력 `[1,128,151]`.
- 본체는 1채널 ResNet18, 마지막 표현 512차원, Linear(11).
- 보조 모델: 10 ms 간격의 특징·측정 가능 마스크 → Conv1D 32/64/64, kernel 5, dilation 1/2/4, GroupNorm, GELU, dropout 0.2.
- ResNet `layer1` 뒤에서 입력 시간 중심 `4*j`에 맞춰 중간 결합. 보조 출력을 주파수 축에 broadcast하고 concat→1×1 projection으로 64채널 복구.
- projection은 본체 identity/보조 zero로 시작한다. 초기 출력이 baseline과 같고, 학습 중 양쪽을 업데이트한다.
- 뒤쪽 `layer2/3/4 → avgpool → 512 → Linear(11)` 유지. `forward_features()`로 향후 head/멀티태스크 실험 가능.
- 임의로 시간을 4구간으로 나누는 pooling 아이디어는 사용하지 않았다.

| 세트 | 스펙트로그램에 추가한 입력 |
|---|---|
| 0 | 없음 |
| 1 | log F0, Δlog F0, 유성 여부·유성 확률 |
| 2 | F1, F2, F3 |
| 3 | 세트 2 + B1, B2, B3 |
| 4 | 세트 2 + ΔF1, ΔF2, ΔF3 |
| 5 | spectral flatness, 정규화 spectral flux, centroid |
| 6 | log RMS |
| 7 | 위 특징 전부, 중복 제거: 17개 + 마스크 17개 |

F0는 pYIN 65~600 Hz, 64 ms 창, 10 ms hop이고 `12*log2(F0/100)`으로 표현한다.
F1~F3/B1~B3는 Praat Burg, 5 formants, 공통 ceiling 5500 Hz, 유효 창 25 ms로 구하며 Hz를 유지한다.
Δ는 연속해서 측정 가능한 5프레임의 국소 기울기다. 연속형 특징은 train의 측정 가능 프레임에서만 z-score를 적합한다.
결측은 표준화 후 0으로 채우고 별도 마스크를 붙인다. 모델에는 NaN/Inf가 들어가지 않는다.

마스크는 **측정값 사용 가능 여부**이며 추정 정확도/포먼트의 물리적 존재 정답이 아니다.
각 분석 창 RMS `<=1e-8` 검사는 현재 PCM16에서 사실상 전체 0인 창 검사다. 작은 목소리를 잘라내는 음량 기준이 아니다.
잡음이 있는 창과 잘못 추정된 유한한 값은 통과할 수 있다. 더 높은 음량 임계값은 도입하지 않았다.
유성 확률은 F0 정확도 확률이 아니다. 이 구분을 사용자와 여러 번 논의했다.

**완료된 첫 실험: `auxiliary-v1`**

0~7 세트 seed42 screening 8회, 세트 0/3/4의 seed43/44 확인 6회,
세트 3 late fusion과 파라미터 수를 맞춘 spectral-only 대조군 2회로 총 16회 학습했다.
주 평가 및 체크포인트 기준은 원본 validation macro F1이다. 학습 증강은 전부 OFF였다.

| 구성 | 3시드 평균 val 정확도 | 3시드 평균 val macro F1 | F1 시드 표준편차 |
|---|---:|---:|---:|
| 세트 0 | 88.34% | 89.26% | 3.65%p |
| 세트 3 | 85.55% | 86.60% | 5.14%p |
| 세트 4 | 88.01% | 89.09% | 3.79%p |

세트 4와 기준의 평균 차이는 -0.17%p에 불과하고 시드별 방향도 달랐다. 개선을 확인하지 못했으며,
일반적인 무용성·통계적 동등성·명확한 열세를 입증한 결과는 아니다. 사전 기준에 따라 세트 0을 선택했다.
세트 7도 실행했다. seed42 val 정확도 88.18%, F1 89.05%였다.

같은 시드의 초기 본체 가중치, 원본 캐시, scaler, 프로토콜이 동일했는지 감사했고 39개 테스트가 통과했다.
세트 3의 late fusion F1은 91.44%, 중간 결합은 92.24%, 용량 대조군은 89.57%였지만 한 시드의 보조 비교다.
추가 파라미터는 약 0.36~0.40%이고 기존 512→11 형태를 유지했다.

소음은 **학습 증강이 아니라 평가 복사본에만** 더했다.
ESC-50 rain/engine/church_bells/keyboard_typing × 추가 SNR 20/10/0 dB, 별도 noise-only 144개다.
Validation은 ESC-50 fold4, test는 fold5의 서로 다른 source ID를 썼다.
향후 학습 잡음에는 이 평가 소스와 겹치지 않는 소스를 사용한다.
SNR은 기존 녹음 full-clip 파워 대 추가 잡음 파워다. Silence/noise-only는 train keyword RMS 중앙값을 기준으로 한 명목 레벨이다.

**Test 사용 이력 — 미공개 상태라고 주장하지 말 것**

- 첫 실험은 선택 결과를 `final_selection.json`에 먼저 고정한 후 세트 0 seed42만 test에서 평가했다.
- 그 모델: val 정확도 91.63%, F1 92.38%; test 148/203, 정확도 72.91%, F1 74.88%.
- 보조 모델들의 test는 평가하지 않았다. 과거 ref 체크포인트의 test도 평가하지 않았다.
- 첫 실험의 test 결과로 해당 실험을 재튜닝하지 않았다. 다만 전체 프로젝트에서 추가 개선을 계속할 계획이었으므로 test를 연 시점이 빨랐다는 점을 사용자에게 인정했다.
- 다음 실험의 선택·튜닝은 test를 제외한 train/validation 또는 개발용 CV에서 한다. 기존 test 점수는 보존한다.
- Test 결과를 근거로 반복 개발한다면 그 test는 개발에 사용된 셈이다. 완전히 독립된 후속 최종 평가를 주장하려면 새 미공개 데이터가 가장 명확하다.
- 사용자는 test 녹음자의 억양/발음이 다른 화자들과 다소 다르다고 설명했다. 이는 사용자 관찰이며 원인으로 검증된 사실은 아니다. 억양·개인차·녹음 환경을 분리하지 못했다. Test 화자를 임의로 제외하거나 그 사람만 겨냥해 조정하지 않는다.

**첫 실험 이후의 추가 진단 — 기존 보고서 해석을 보완할 내용**

- 진단 데이터: `runs/auxiliary-v1/diagnostics/`에 보관. 이 경로는 Git에서 제외되므로 같은 작업공간에만 있을 수 있다.
- `ref_train_validation.json`: ref 원래 체크포인트를 eval 모드로, 증강·가중 샘플링 없이 고유 train 609개 전체와 val 203개에서 평가했다.
- Ref train 정확도/F1 100%, loss 0.002346. Val 정확도 92.12%, F1 92.58%, loss 0.277413.
- 현재 환경에서 재평가한 ref validation 예측은 저장된 원래 203개 예측과 모두 같았다. 원래 ref 학습 torch는 2.3.1+cu121이었다.
- 이번 실험의 기준 모델도 세 시드 모두 train 100%. Val 정확도는 seed42/43/44 각각 91.63/89.16/84.24%.
- 전체 진단 대상 14개 일반 모델 중 13개는 train 100%, 나머지 하나도 99.84%였다.
- **과적합 경향은 의심할 수 있으나 이를 보조 실험 부진의 주원인이라고 단정한 설명은 수정했다.** 기준부터 있는 격차이고, 화자/환경 차이도 섞일 수 있다.
- Ref 마지막 4 epochs의 val 정확도는 86.70→88.18→88.67→92.12%로 개선됐다. Train 100%만으로 추가 학습이 무의미하거나 심한 과적합이 확정됐다고 말하지 않는다. Val 상승도 과적합 부재의 증명은 아니다.
- `inference_interventions.csv`: 재학습 없이 보조 경로 제거·입력 0·마스크만 유지·클립 섞기·시간 뒤집기를 검사했다. 원래 val F1도 재현했다.
- 예: 세트 4 seed42는 보조 projection을 끄면 val F1 90.94→85.69%. 보조 경로를 실제 사용하는 것은 확인했다.
- 이 추론 조작은 학습과 다른 입력 분포를 만들 수 있다. 새로 학습한 기준 모델과 동등한 대조군이 아니며 보조 정보의 독립적 성능 이득을 증명하지 않는다.
- `f0_by_speaker.json`: F0를 채택한 프레임의 중앙값은 spk02/03/04/05 각각 약 244/110/129/95 Hz.
- Val의 채택된 F0 프레임 중 약 39.5%가 65~67 Hz에 있었고 이 프레임들의 유성 확률 중앙값은 약 0.01이었다. 추출 하한에 몰린 값의 품질 점검이 필요하다는 단서다. 모두 오측정이라는 뜻은 아니다.
- F0가 없는 포먼트 세트까지 F0 문제 하나로 설명할 수는 없다.
- 임시 진단 스크립트는 `/tmp/audio_aux_diagnose.py`. 필요하면 재사용 전에 경로·내용을 읽고, 재현할 코드는 정식 실험 폴더에 보관한다.

**Ref의 오답 쌍**

Ref는 val 203개 중 16개 오답이며 모두 spk05다. 후진 20개 중 7개를 틀려 가장 많은 오류를 보였다.
정답→예측: 후진→시작 3, 후진→정지 3, 오른쪽→왼쪽 2,
후진→재시작·시작→재시작·재시작→시작·복귀→대기·전진→복귀·전진→정지·정지→대기·출발→재시작 각 1.
파일 목록은 `reports/task2-fixed-1p5-v1/validation/error_analysis.csv`.

**다음 실험에서 논의한 방향**

1. Test spk01을 제외한 spk02~05에서 화자별 4-fold 개발 CV를 구성한다.
2. 먼저 세트 0과 세트 4로 증강 없는 CV 기준을 구한다.
3. 증강 후보를 각각 비교한 뒤 유효한 것만 결합한다.
4. Weight decay·본체 Dropout 등을 별도로 비교한다. 앞단 고정·학습률 분리는 후속 후보다.
5. 공통 학습 조건을 선택한 뒤 0~7 세트를 다시 비교한다. 세트 7 전체 결합도 포함한다.
6. 처음에는 모델 2개 × fold 4개 × seed 1개 = 조건당 8회로 선별하고, 유망 조건은 seed42/43/44로 확인하는 방안을 제안했다.

| Fold | Train | Validation |
|---|---|---|
| 1 | spk03/04/05 | spk02 |
| 2 | spk02/04/05 | spk03 |
| 3 | spk02/03/05 | spk04 |
| 4 | spk02/03/04 | spk05 |

각 fold는 새 모델/optimizer/scheduler/선택 기록을 사용한다. 기존 fine-tuned 모델에는 다른 fold의 validation 화자가 들어 있으므로 CV 초기값으로 재사용하지 않는다.
각 fold train에서 scaler와 학습 데이터 기반 참조값을 다시 적합한다. 고정 추출기를 사용한 원시 특징 캐시는 재사용 가능하지만 기존 scaler를 통째로 공유하지 않는다.
주 지표는 원본 CV 평균 macro F1, 보조 지표는 화자별 F1·최저 화자 F1·시드 변동·오류 쌍이다.
학습 seed 반복과 검증 화자 변경은 구분해서 기록한다. 네 화자의 CV를 독립적인 수백 명 평가처럼 해석하지 않는다.
CV 결과로 설정/epoch를 선택한 점수는 개발용 점수이며, 독립적인 최종 test 성능은 아니다.

**증강 시작 후보 — 아직 적용하지 않음**

| 후보 | 제안 범위/주의 |
|---|---|
| 음량 | ±3~6 dB, clipping 방지 |
| 시간 이동 | 남는 무음 범위에서 이동. 원형 이동 대신 padding 사용, 발화 보존 |
| 속도 | 피치를 유지하는 0.9~1.1배부터. 단어를 자르지 않고 1.5초 입력 유지 |
| 피치 | 포먼트 보존 옵션으로 ±1~2반음부터, 청취 점검. 새 억양 전체를 재현하는 방법은 아님 |
| 실제 배경음 | SNR 10~30 dB 정도의 약한 조건부터. 평가 잡음 소스와 분리 |
| 잔향/EQ | 약한 방·마이크 변화부터 |
| SpecAugment | 작은 시간·주파수 마스크부터. 단어 구별 단서 전체를 가리지 않기 |

기존 `src/dataset.py`에는 gain·circular shift·단순 잡음·time/frequency masking이 있지만 첫 실험에서는 OFF였다.
기존 옵션을 켜는 것만으로 보조 모델 증강이 구현되는 것은 아니다. 현재 auxiliary의 학습 캐시는 고정 입력이며 augmentation을 거부한다.
파형 변형은 **증강 WAV → spec와 보조 특징 모두 재추출** 순서로 구현한다. 시간 중심도 계속 맞춘다.
추출 후의 SpecAugment 같은 입력 마스킹은 별도 정책을 명시한다. 증강 원본/파생본은 동일 fold train 안에만 둔다.
원본 validation이 주 평가다. 소음 스트레스 평가를 유지한다면 모든 모델에 같은 별도 조건을 쓴다.

**과적합 완화의 다른 후보 — 모두 비교 대상이지 효과가 확정된 설정은 아님**

- Weight decay: 현재 1e-4 → 1e-3 등 비교.
- 본체 분류기 앞 Dropout: 현재 없음 → 0.2부터 비교. 보조 경로에는 이미 0.2가 있다.
- Label smoothing: 0.05 등 작은 값부터.
- 본체 초반 층 고정: 해당 BatchNorm running 통계도 관리.
- 본체/새 head·보조 경로 학습률 분리: 예시 5e-5 / 5e-4. 학습 안정화 후보이며 낮은 LR 자체가 과적합 해결을 보장하지 않는다.
- 보조 CNN 폭 축소, 보조 경로 dropout.
- 조기 종료/최고 val 체크포인트 선택. 최고 체크포인트는 이미 선택 중이며 epoch를 무작정 줄이는 것을 우선하지 않는다.
- SWA/가중치 평균화, 향후 멀티태스크 보조 과제.
- Mixup은 후보지만 두 음성을 섞었을 때 F0·포먼트와 soft label 처리를 설계해야 하므로 후순위.
- 다양한 화자·세션 및 학습용 배경음 확보. 기존 train silence는 9개뿐이다.

모든 조건을 한꺼번에 변경하지 않는다. 기준과 보조 모델에 같은 조건을 적용해 추가 입력의 효과를 구분한다.
성공 기준은 **원본 CV 성능과 화자별 안정성 개선**이다. Train 점수만 낮아져 gap이 줄어드는 것을 개선으로 판단하지 않는다.

**다음 작업자가 먼저 읽을 파일과 실행 주의점**

- `reports/auxiliary-v1/README.md`, `interpretation.md`: 완료된 결과와 기존 해석.
- `experiments/auxiliary/README.md`: 추출·정렬·모델·프로토콜 정의.
- `experiments/auxiliary/features.py`, `model.py`, `data.py`, `study.py`, `report.py`.
- `reports/auxiliary-v1/all_runs.csv`, `seed_summary.csv`, `fairness_audit.csv`, `execution_audit.json`.
- `reports/auxiliary-v1/validation_case_review.md`: validation 오류 20개. 청취 원인은 미확정이다.
- Ref 결과: `reports/task2-fixed-1p5-v1/`. 체크포인트: `runs/task2-fixed-1p5-v1/best.pt`.
- 첫 실험 체크포인트·캐시: `runs/auxiliary-v1/`. 원본 train+val 캐시에는 812개만 들어 있다.
- 현재 `study.py`는 고정 split 실험용이며 CV 명령이 없다. CV 구현을 추가하거나 별도 모듈을 만들어야 한다.
- 첫 실험의 `protocol.json`은 동결되어 있다. CV/증강을 기존 `runs/auxiliary-v1`에 섞지 말고 새 study/report 폴더를 사용한다. 예: `runs/auxiliary-cv-v1`, `reports/auxiliary-cv-v1`.
- 기존 `finalize`는 test 평가를 실행한다. 다음 개발용 CV에서 자동 호출하지 않는다.

```bash
cd /root/audio
git status --short --branch
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

이 문서 작성 시점의 코드 검증 결과는 39 tests passed다. 새 CV/증강 코드에는 분할 누출 방지,
fold별 train-only 표준화, 시간 정렬, 증강된 WAV/보조 특징 일치처럼 실제 오류를 잡는 검증을 추가한다.
현재 완료된 16회 실험을 반복 실행할 필요는 없다.

공식/연구 참고: [그룹 CV](https://scikit-learn.org/stable/modules/cross_validation.html#leave-one-group-out),
[음성 증강](https://docs.pytorch.org/audio/stable/tutorials/audio_data_augmentation_tutorial.html),
[SpecAugment](https://www.isca-archive.org/interspeech_2019/park19e_interspeech.html),
[Rubber Band 포먼트 보존](https://breakfastquay.com/rubberband/code-doc/classRubberBand_1_1RubberBandStretcher.html),
[Dropout](https://jmlr.org/papers/v15/srivastava14a.html),
[AdamW](https://arxiv.org/abs/1711.05101), [Mixup](https://arxiv.org/abs/1710.09412),
[SWA](https://arxiv.org/abs/1803.05407).

**2026-11-11 추가 — 원본 주변음으로 1.5초 채우기 조사**

- 사용자는 파일 안에서 변하는 피치 증강에 관심이 있고, 1.5초 전체를 실제 녹음 소리로
  채우기를 원한다고 밝혔다. CV/증강 학습을 실행하기 전에 데이터 처리 방향을 논의 중이다.
- 더미 노트북은 원본 전체를 읽고, 원본이 충분하면 1.5초를 crop하며 짧을 때만 zero pad한다.
  기존 개인 VAD 전처리는 원본에 여유가 있어도 발화 주변만 남긴 뒤 zero pad하므로 차이가 있다.
  PDF에서 모든 파일을 실제 배경음으로 채워야 한다는 필수 조건은 확인하지 못했다.
- 원본 1,015개 중 350개는 파일 자체가 1.5초 미만(약 1.067~1.493초)이다. 화자별로
  spk01 79개, spk04 118개, spk05 153개이며 모두 keyword다. 단순 제외하면 평가 구성이 달라진다.
- 나머지 665개를 `processed_datasets/real-context-1p5-candidates-v1/`에 후보로 만들었다.
  기존 검수 crop 전체를 포함한 원본 연속 24,000샘플이며 인공 패딩/반복/속도 변환은 없다.
  633개는 주변 녹음이 늘었고 32개(긴 단어 17개+silence 15개)는 기존 파일 해시도 동일하다.
- 새 주변음 때문에 `대기/spk04_003`, `출발/spk05_174`의 전체 gain을 낮췄다. 두 파일은
  피크 관련 검수 대상으로 기록했으며, 후보본 청취 검수는 아직 하지 않았다.
- **짧은 350개의 처리는 미정이다.** 같은 파일의 비발화 배경음 반복, 기존 zero pad 유지,
  재녹음 대상으로 표시하는 선택지를 설명했고 사용자는 장단점을 물었다. 특정 방식을 아직
  선택하지 않았으므로 배경음 반복을 이미 승인했다고 해석하지 않는다.
- 현재 후보본은 완성된 학습 데이터셋이 아니다. `candidate_metadata.csv`만 있고 전체용
  `metadata.csv`는 없다. 기존 학습 설정·기존 데이터·기존 실험은 변경하지 않았다.
- 재현 코드: `preprocessing/real_context.py`. 조사/CSV/감사 기록:
  `reports/real-context-1p5-v1/`. 원본·검수 WAV 2,030개, 원본 M4A 1,015개와 ZIP의 기록된 해시,
  후보 665개의 원본 구간 대비 PCM 샘플 일치를 확인했다. 신규 5개 포함 **44 tests passed**.
  학습·CV·새 특징 추출은 수행하지 않았다.

**2026-11-11 후속 확정 — 짧은 원본은 부족한 길이만 0 패딩**

- 사용자가 원본이 충분한 665개는 실제 연속 1.5초, 짧은 350개는 원본 전체를 보존하고
  부족한 길이만 좌우 0 패딩하는 방식을 승인했다. 앞의 "정책 미정" 기록은 이 결정으로 해소됐다.
- 전체 1,015개를 `datasets/task2/real-context-1p5-v1/`에 별도로 생성했다. 라벨·화자·기존
  train/val/test 609/203/203은 동일하다. 기존 원본과 `fixed-1p5-v1`은 그대로 보존했다.
- 새 설정: `configs/task2_real_context.json`. 기존 `task2_real.json`은 이전 기준을 계속 가리킨다.
  새 설정도 현재는 기존 단일 분할이며 4-fold CV를 자동으로 실행하는 설정이 아니다.
- 코드: `preprocessing/real_context.py --short-policy zero`. 전체 metadata/recipe/manifest를 생성하며
  기존 `preprocessing.clip_audio --recipe`로 1,015개를 재현했을 때 모든 출력 해시가 일치했다.
- 새 원본 주변음을 포함하면서 전체 gain을 낮춘 파일은 6개다. 기존 후보 2개에
  `대기/spk04_008`, `대기/spk05_015`, `오른쪽/spk05_076`, `전진/spk05_143`이 추가됐다.
  청취 검수는 아직 하지 않았다. 전체 목록·수치는 `reports/real-context-1p5-v1/full_dataset_audit.json`.
- 전체 **47 tests passed**. 라벨/분할·모든 원본 WAV/기존 WAV 해시·추가 패딩의 0값·원본 전체
  보존·새 recipe 재현을 확인했다. CV·학습·특징 재추출은 아직 구현/실행하지 않았다.
- 다음 실험은 이 새 데이터에서 세트 0·4의 무증강 4-fold CV → 피치/속도/잡음/마스킹 개별 비교
  → 유효 증강 결합 → weight decay/본체 dropout 개별 비교·유효 조합 확인 → 필요 시 앞단 고정/LR 분리
  → 선택한 공통 조건에서 0~7 전체 비교 순서다. test spk01은 개발용 선택에서 제외한다.

**2026-11-11 후속 실행 — CV 실험 시작**

- 사용자가 위 전체 플로우의 실험 실행을 승인했다. 이후 네트워크 변경으로 연결이 끊겼다가
  "여기서 재개"를 요청했다. 최초 nohup 준비 프로세스는 종료됐고 완성 캐시가 없었으므로,
  파일별 row cache를 저장하도록 보완하여 원본 개발용 812개 특징 추출을 완료했다.
- 구현: `experiments/auxiliary_cv/`. 실행:
  `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u -m experiments.auxiliary_cv.pipeline`.
  현재 tool 실행 session ID는 25317이며, 실제 상태는 `runs/auxiliary-cv-v1/active.json`,
  로그 `driver.log`, 완료 run `runs/*/summary.json`, 보고서 `reports/auxiliary-cv-v1/`에서 확인한다.
  프로세스가 종료됐으면 같은 명령으로 재개한다. 완료 fold와 특징 row는 재사용하며 epoch checkpoint도 복구한다.
- 전체 55 tests passed. 실제 GPU에서 2 epoch(set4, seed99, spk02)의 연속 학습과
  1 epoch 후 강제 중단→재개가 가중치·검증 결과까지 일치했다. 이 4 epoch는 엔지니어링 검증이며
  조건 선택 실험에 사용하지 않는다. `runs/auxiliary-cv-v1/recovery_check.json`에 기록했다.
- 사전 프로토콜은 `runs/auxiliary-cv-v1/protocol.json`에 코드 해시와 함께 고정됐다.
  실행 중 Python 코드를 바꾸면 같은 study에서 조용히 재개하지 않는다.
- Ubuntu Jammy Rubber Band 2.0 R2/FFTW/libsamplerate를 study/runtime에 풀어 사용한다.
  학습 가상환경 수치 패키지는 재설치하지 않았다. 처음 발견한 피치 32ms 앞당김은
  공식 CLI의 입력 padding+출력 latency 보정으로 수정했고 합성 impulse에서 오차 0을 확인했다.
- 학습 잡음 ESC-50 16개는 기존 평가 source ID와 분리했다. 파형은 클립당 고정 2개 변형을
  모델/seed 사이에 공유하며 draw마다 정책을 독립 확률 0.5로 적용한다. 검증은 원본만 쓴다.
  잡음 SNR은 현재 원본 full-clip 파워 대 추가 잡음이며 silence도 자신의 원본 파워를 쓴다.
- 피크 예외 6개는 원본 구간 보존·주변 peak 수치를 확인했으나 사람의 청취는 아직 하지 않았다.
  `preflight.json`에 이 한계를 기록했다. Test spk01의 특징 추출과 모델 평가를 하지 않는다.
- 이전 실험 업로드도 GitHub 실시간 확인했다. 원격 `study/jaehun`은 ed58e34이며 16회 코드와
  보고서가 포함된다. 새 데이터/CV 코드는 아직 로컬 변경이며 push하지 않았다.

**CV 진행 확인 및 최종 보고 명령**

- 공식 학습은 현재 `experiments.auxiliary_cv.pipeline`에서 진행 중이다. 같은 driver를 중복 실행하지 말고
  `runs/auxiliary-cv-v1/active.json`, `driver.log`, 실행 tool session 25317의 상태부터 확인한다.
- 무증강 3시드 결과: set0 평균 CV F1 76.7580%, set4 80.3167%.
  개별 증강의 seed42 선별에서는 속도와 마스킹이 후보로 선택됐고, 마스킹의 3시드 확인 평균
  (set0/4 통합)은 약 80.40%로 무증강 78.54%보다 개선됐다. 최종 조건 선택은 아직 진행 중이다.
- 파이프라인 완료 후 `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.report_auxiliary_cv`
  를 실행한다. 이 별도 보고 스크립트는 frozen training 코드 밖에 있으며, 예측·최고 epoch·fold·scaler·source hash·
  초기값과 원본/기존/새 WAV 총 3,045개 해시를 다시 검증하고 그래프·오류 쌍·최종 요약을 생성한다.
- 그래프 2개를 `view_image`로 확인하고 실제 결과를 한국어로 설명한다. 학습 코드는 source hash가 고정됐으므로
  지금 내용을 바꾸지 않는다. 보고 스크립트의 오류 수정은 학습 프로토콜에 영향을 주지 않는다.

**2026-11-12 사용자 요청 — 서버 재부팅 전 정지**

- 사용자가 서버 재부팅을 위해 실험 정지를 명시했다. 완료 학습 126회와 best checkpoint를 보존했다. 재개는 사용자가 재부팅 완료 후 요청하면 진행한다.
- 재개 시도는 GPU 접근 오류로 종료됐다. RTX 3090 정보는 보이나 `/dev/nvidiactl`, `/dev/nvidia3`, `/dev/nvidia-uvm` open이 EPERM이며, read-only devices cgroup 허용 목록에 NVIDIA 장치가 없다. `nvidia-smi`도 NVML Unknown Error. 측정은 `environment_blocker.json`에 기록했다.
- 현재 증강 선택은 mask+speed이며 3시드 set0 F1 82.5086%, set4 83.4168%. Weight decay 1e-3의 seed42 비교 8회는 완료했고, dropout 0.2는 6/8회 완료했다. 다음 미완료 run은 `c75e815312093_set4_seed42_spk04`다. 진행 중 epoch checkpoint는 없으며 완료 run은 반복하지 않는다.
- 재부팅 후 `nvidia-smi`와 Torch CUDA 초기화를 먼저 확인한다. 이후 기존 pipeline 명령을 동일하게 재실행하고 driver.log는 `>>`로 추가한다. frozen Python 소스 19개는 변경하지 않는다. 최근 시도 tool session 75287은 GPU 오류로 종료됐다.
- 독립 보고 스크립트에 `--partial`을 추가했으나 사용자의 정지 요청으로 아직 실행하지 않았다. 전체 종료 후 최종 audit/그림/오류 분석은 여전히 필요하다. 새 코드/데이터/보고서는 로컬 변경이며 commit/push하지 않았다.

**2026-11-12T09:19:38+09:00 사용자 요청으로 재개**

- 재부팅 후 RTX 3090 / 드라이버 550.107.02, Torch CUDA 실제 연산 정상 확인. 기존 실패 기록은 previous_gpu_failure.json으로 보존하고 environment_blocker.json은 resolved_after_reboot로 표시했다.
- pipeline을 기존 frozen 소스로 재개했다. tool session 53468, 로그 driver.log에 append. 중복 실행하지 않는다.
- 사용자용 진행 정리는 reports/auxiliary-cv-v1/INTERIM.md. 완료 시 최종 결과와 남는 한계로 갱신한다.

**2026-11-12T09:34:22+09:00 다음 실험 전체 완료**

- GPU 재부팅 복구 후 남은 비교를 완료했다. 정식 학습 184회/1,840 epochs, 별도 복구 검증 4 epochs. pipeline session 53468과 최종 보고 session 30143 모두 exit 0. 실행 중 학습 프로세스 없음.
- 공통 조건: mask+speed, weight decay 1e-4, Dropout 0, 앞단 학습, 본체 LR 비율 1. pitch/noise, weight decay 1e-3, Dropout 0.2, 앞단 고정, LR 0.1배는 사전 기준 미충족. 조건과 source hash는 변경하지 않았다.
- 0~7 seed42 전체 CV 후 상위 세트2/3과 기준0/4를 3시드 확인했다. 최종 F1: set3 84.6233%, set2 83.8237%, set4 83.4168%, set0 82.5086%. 세트3 선택, 기준 대비 +2.1147%p. 최저 화자 3시드 평균 67.0645%→72.3427%. 독립 최종 평가가 아닌 개발 CV이며 spk01을 사용하지 않았다.
- 실행 184개의 예측·지표·최고 epoch·checkpoint/설정/소스 일치·분할·표준화·초기값과 최종 선택 재계산을 감사했다. 원본/이전/새 WAV 3,045개 해시와 512차원 분류 전 표현 보존 검사 통과. 학습 전 테스트 55개 통과, 이후 frozen source 변경 없음.
- 최종 해석: reports/auxiliary-cv-v1/interpretation.md. 사용자가 안내받은 INTERIM.md도 완료 상태로 갱신했다. execution_audit.json, run_audit.csv(가중치 SHA 포함), paired_predictions.csv, error_pairs.csv, persistent_errors.csv에 근거. feature_sets_cv.png와 speaker_stability.png를 화면 확인했다.
- 속도 고정 변형 1,600개: 가속+0 패딩 797, 검수 구간 보존 감속 784, 감속 제외 19. silence 변형 24는 유지. 최대 추가 패딩 135.8125ms. 증강 적용 감사에 기록했으며 실제 학습 draw 횟수와 구별한다.
- 남는 한계: 개발 화자4명, 조건/epoch 선택에 같은 CV 사용, 유한 파형 변형2개, 화자별 padding/녹음환경 차이, F0/포먼트 추정 품질, 피크 예외6개 청취 미실시. 새 잡음/거리/공간의 강건성을 별도로 증명한 실험은 아니다.
- 가중치·캐시는 runs/auxiliary-cv-v1에 로컬 보관. 새 코드·데이터·보고서는 아직 commit/push하지 않았다. 이전 16회 실험 원격 확인 ed58e34 기록은 이전 확인 시점의 결과다.

**2026-11-12T10:26:25+09:00 사용자 요청 — 파일별 epoch 악화와 화자1/3/4 조사 완료**

- 사용자가 최종 세트3의 spk02 seed44에서 초반 정답→후반 오답 파일과 별도 spk01/03/04 조사를 요청했다. 이는 spk01을 별도 진단하는 명시적 요청이며 기존 CV 조건/선택은 변경하지 않았다.
- 원래 epoch별 예측은 저장되지 않아 experiments/diagnose_auxiliary_cv_files.py로 spk02/03/04의 seed44를 별도 study runs/auxiliary-cv-file-diagnostics-v1에서 각10epoch 재현했다. 원래 history CSV·최적 가중치 텐서·최적 예측 CSV·지표가 세 실행 모두 동일했다. 진단 재현30epochs를 정식184회 결과와 섞지 않는다.
- 각 epoch의 203개 파일 예측을 reports/auxiliary-cv-v1/file_diagnostics에 보존했다. spk02 모델 epoch별 가중치를 보관하고, 사용자 요청 spk01의203개를 별도 원본 특징 추출한 후 학습spk03/04/05에서 적합한 원래 spk02-fold scaler로만 평가했다. spk01 학습/표준화 적합/설정 선택 없음.
- 공통 epoch3→10 기준: spk02 F1 70.14→57.26%, 정답136→110, 정답→오답43/회복17. spk01(같은spk02모델 외부평가)71.86→64.12%, 정답140→133, 악화24/회복17. spk03(고유미학습fold)72.02→85.61%, 악화10/회복33. spk04(고유미학습fold)75.78→86.23%, 악화10/회복34.
- spk03의 원래 최고epoch7→10 기준 악화7개, spk04 최고epoch9→10 기준 악화4개도 별도 CSV/목록으로 구분했다. 3/4는 epoch3→10 전체 성능이 개선돼 spk02와 같은 전체 악화 경향은 아니다.
- 목록 생성/검증: experiments/report_auxiliary_cv_file_diagnostics.py. Markdown에 모든 악화 WAV 링크·정답/나중예측·원본 파일명, CSV에10epoch 예측/정답확률/NLL과패딩/gain을 보존. 총8,120개 예측에서 지표와 NLL 복원 검증. 사람의 청취 검수/원인 판정 없음.
- spk02 악화43개는 원본 부족분의zero pad가0개다. 복귀18/20→8/20, 정지15→8, 후진17→10으로 하락하고 대기10→16, 출발13→19로 일부단어는 개선. 파일명 거리20cm/50cm와prefix별 분포를 기록했으나 원인으로 확정하지 않는다.
- 주 진입점 reports/auxiliary-cv-v1/file_diagnostics/README.md, spk02.md, spk01.md, spk03.md, spk04.md. speaker_epoch_curves.png 화면 확인. 원래 보고서에는 후속 spk01 진단 범위를 별도로 안내했다.

**2026-11-12T10:35+09:00 사용자 요청 — 보조 제외 모델과 같은 파일별 진단 비교 완료**

- 사용자가 직전 세트3 진단을 보조 없는 모델과 같은 방식으로 비교하도록 요청했다. 공통 mask+speed·weight decay1e-4·Dropout0·seed44·화자 분할을 유지하고 세트0만 비교했다. 무증강 기준과 혼동하지 않는다.
- experiments/diagnose_auxiliary_cv_files.py에 --set-id 0을 추가했다. frozen 학습19개 소스는 그대로다. runs/auxiliary-cv-file-diagnostics-set0-v1에서 spk02/03/04를 각10epoch 재현했고 원래 history CSV·최적 가중치 텐서·최적 예측·점수 모두 일치했다. 세트0 진단30epochs 추가, 세트3와 합계60epochs이며 원래 정식184회는 변경하지 않았다.
- spk01은 세트0 spk02-fold epoch별 모델을 별도 평가했다. 기존 spk01 원본 특징 캐시를 재사용하고 동일 원래 train scaler를 적용했다. spk01 학습·scaler 적합·선택은 하지 않았다.
- experiments/compare_auxiliary_cv_file_diagnostics.py로 세트0 진단 보고서 및 file_diagnostics_comparison을 생성했다. 두 세트·4화자·10epoch의16,240예측에서 정확도/F1/NLL·동일파일·라벨·전처리 일치를 검증했다. speaker_epoch_comparison.png 화면 확인. CSV/Markdown에203개씩 양쪽10epoch 예측·정답확률·NLL·원본파일명·패딩과 악화/회복·단어별 정답 수를 제공한다.
- spk02: 세트0 F1 50.13→52.55%, 정답99→104, 악화30/회복35. 세트3 70.14→57.26%, 정답136→110, 악화43/회복17. 둘 다 epoch3 정답83개에서 악화는24(28.9%) 대23(27.7%). 앞서 세트3 악화43개 중 세트0도 악화14개, 정답유지9개, 오답유지19개, 회복1개. 세트0 최고epoch4 62.82→epoch10 52.55%로 역시 후반 하락한다.
- spk01(외부): 세트0 56.36→65.23%, 악화19/회복37 대 세트3 71.86→64.12%, 악화24/회복17. spk03:66.33→84.22%,6/47 대72.02→85.61%,10/33. spk04:76.55→84.17%,7/28 대75.78→86.23%,10/34.
- spk02 epoch10 세트3만 정답25개·세트0만 정답19개로 포함쪽 정답6개 많다. 반면 평균loss는 세트0 1.6870·세트3 1.8016으로 세트3이 나쁘다. F1과 확신 변화는 구분한다. 초반 점수·학습 진척 차이가 크므로 악화 개수만으로 보조 특징의 인과를 확정하지 않는다. 한 시드 사후 진단이며 청취/라벨 검수는 하지 않았다.
- 진입점 reports/auxiliary-cv-v1/file_diagnostics_comparison/README.md와 spk02.md(앞선43개 양쪽 예측·WAV 링크), spk01.md/spk03.md/spk04.md. 세트0 개별원자료는 file_diagnostics_set0. 새 파일 로컬 보존, commit/push 안 함.

**2026-11-12T10:44+09:00 사용자 요청 — 최고 성능 체크포인트끼리 비교 완료**

- 직전과 동일한 mask+speed·seed44·세트0/3에서 각자 원래 검증 Macro F1로 선택한 최고 epoch의 저장 예측을 비교했다. experiments/compare_auxiliary_cv_best_checkpoints.py, 보고서 reports/auxiliary-cv-v1/best_checkpoint_comparison/README.md. 추가 학습0epochs, 정식184회/원래 선택/frozen소스는 그대로다.
- spk02: set0 epoch4 62.8209%, 정답126/203, loss1.2559 대 set3 epoch3 70.1438%, 정답136, loss1.1441. 보조 포함 F1 +7.3229%p. 포함만 정답36개·없음만 정답26개·둘 다 정답100개·둘 다 오답41개.
- spk03: set0 epoch8 87.0179% 대 set3 epoch7 87.5904%(+0.5725%p), 정답175→176, 포함만13/없음만12. loss는0.4534→0.5132로 포함쪽이 나쁘다. spk04: 양쪽epoch9,84.7280→87.4249%(+2.6969%p), 정답173→180, 포함만12/없음만5.
- spk01은 자기 결과로 epoch를 고르지 않았다. spk02에서 선택한 set0 epoch4/set3 epoch3을 그대로 평가해58.4589→71.8616%(+13.4027%p), 정답115→140, 포함만40/없음만15, loss1.9459→1.1088. 이 행을 CV 평균에 합치지 않는다.
- spk02 단어별 정답(/20): 복귀2→18, 오른쪽11→18, 정지10→15 개선. 출발20→13, 대기16→10, 재시작17→12 감소. 전체 우위가 모든 단어의 우위를 뜻하지 않는다. 앞서 epoch3→10에 악화한43개도 양쪽 최고점에서 별도 대조CSV/WAV목록으로 제공한다.
- 두 세트×4화자=1,624개 최고점 예측에서 정확도/F1/NLL을 재계산했다. 각 선택 epoch가 원래 history의 최초 최고F1인지 확인했고, spk02/03/04의 선택 epoch 예측은 원래 best checkpoint 예측과 정확히 일치했다. 원래 checkpoint·예측·새 산출물 해시와 WAV/문서 링크 검증. 이전 epoch별 비교 산출물40개의 해시도 그대로다.
- 각 화자 문서에 세트3만 정답·세트0만 정답·둘 다 오답인 전체 파일과 양쪽 예측·실제WAV 링크, CSV에 전체203개·정답확률/NLL·원본명/패딩. 한 시드의 개발 검증 최고점이며 독립test나 오류 원인의 인과 판정이 아니다. 청취/라벨 검수 안 함. 새 자료는 로컬 보존, commit/push 안 함.

**2026-11-12T10:56+09:00 사용자 요청 — 총합 기준 공통 최고 epoch 비교 완료**

- 사용자가 각 화자의 최고 epoch를 따로 고르는 대신 총합 최고점을 요청했다. 세트마다3시드(42/43/44)×4-fold(spk02~05)의 평균 Macro F1이 최고인 공통 epoch 하나를 골라 모든 화자·시드에 동일하게 적용했다. spk01은 epoch 선택·총합에서 제외했다. CV의 학습 구성이 다른 fold모델4개를 공통 시점으로 비교한 것이며 단일 전체학습 모델을 새로 만든 작업은 아니다.
- 원래24개 history/240행에서 set0 epoch8 평균F1 79.9893%, 정확도79.7209%, loss0.6937 대 set3 epoch9 F1 82.3046%, 정확도81.5271%, loss0.6731로 계산했다. F1 +2.3152%p. 같은 공통epoch의 시드별F1차이는 seed42 +1.8512,43 +4.5697,44 +0.5249%p. 앞서 화자별 최고epoch로 계산한82.5086/84.6233%와 기준이 다르다.
- 같은epoch 직접비교: epoch8 set0 79.99/set3 81.58%; epoch9 set0 79.66/set3 82.30%. 공통선택epoch에서 각화자3시드평균 F1 set0→3: spk02 63.14→68.05,03 85.75→85.87,04 83.82→86.08,05 87.25→89.21%.
- seed44 파일 비교를 완성하려고 spk05 set0/3의 각10epoch만 별도 재현했다. 기존 experiments/diagnose_auxiliary_cv_files.py의 replay를 재사용했고 원래 history·최적 모델 텐서·최적 예측·지표 모두 일치했다. 이번20epoch 추가, 파일 진단 누적80epoch. 정식184회/1,840epoch 및 frozen소스19개는 그대로다.
- experiments/compare_auxiliary_cv_common_epoch.py로 reports/auxiliary-cv-v1/common_epoch_comparison을 생성했다. 원래 전체history, epoch별전체평균/시드별평균, 선택epoch24개fold/seed지표, 3시드화자별평균과seed44의812개짝비교·화자별전체WAV목록을 저장했다. 공통선택epoch의 총2,030개예측(두세트×CV812+별도203)에서 정확도/F1/NLL을 재계산했다.
- seed44의 공통epoch8/9 평균CVF1 78.6363→79.1612%;812개를 합친F1 78.9200→79.0008%. 정확도78.2020→77.8325%,정답635→632,loss0.7235→0.8600. 세트3만정답59개·세트0만정답62개로, 이seed에서는 F1과정답수/loss의방향이다르다. 합산F1/정확도로epoch를다시고르지않았다.
- seed44 화자별F1 set0→3: spk02 55.13→59.10,03 87.02→86.47,04 84.02→87.42,05 88.38→83.65%. 별도spk01은 동일spk02-fold모델 epoch8/9에서67.00→62.18%,정답135→130,포함만정답11/없음만16. spk01의최고epoch를고르지않았다.
- common_epoch_curves.png/.svg에 전체F1/loss와시드별CV평균의SD, 선택시점별표시를그렸다. 화면확인완료. 새산출물46개·입력34개해시와358개WAV/보고서링크검증완료. 직전epoch비교40개·개별최고점비교31개산출물해시도모두보존했다.
- 진입점 common_epoch_comparison/README.md, 각spkXX.md, seed44_all_812_paired.csv. 개발CV의사후선택이며 독립test/인과판정/청취검수결과가아니다. 새자료로컬보존,commit/push안함.
