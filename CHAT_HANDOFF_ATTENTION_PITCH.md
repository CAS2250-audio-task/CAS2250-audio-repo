# 다음 채팅 인계: 한국어 짧은 명령 음성 분류 / 0dB 강건성 실험

## 현재 상태와 사용자 의도

최근 요청은 완료된 실험 결과 확인 후, 다른 채팅에서 이어갈 수 있게 정리하는 것이다.
아래 실험은 모두 완료됐다. 재개할 미완료 fold는 없고, 다음 학습 실험은 아직 정하지 않았다.
새 채팅에서는 이 파일과 실제 결과 파일을 먼저 읽고 사용자와 다음 방향을 논의한다.

장기 목표는 깨끗한 음성의 높은 성능을 유지하면서 다양한 소음의 0dB 조건에서 화자 일반화를 개선하는 것(사용자 희망 약95% 이상)이다.
사용자는 특히 화자02의 낮은 성능, 발화 전후 배경에 대한 불필요한 attention, 발화 내부의 다양한 피치, 모델 경량화에 관심이 있다.
준비 및 학습 ETA, SSH/VSCode 연결을 종료해도 되는 시점을 알려주는 것을 중요하게 생각한다.

## 작업 폴더와 브랜치

| 용도 | 경로 | 브랜치 / 현재 HEAD |
|---|---|---|
| 기본 checkout | `/root/audio` | `jaehun` / `cb9dd65` |
| 최신 완료 실험 | `/root/audio/.worktrees/attention-pitch` | `attention-pitch-v1` / `6272cfd` |
| 직전 정규화·증강·보조 실험 | `/root/audio/.worktrees/overnight` | `overnight-regularization-formant-aux` / `58b6849` |
| 이전 Conformer 실험·환경 | `/root/audio/.worktrees/pretrain` | `pretrain` / `9fbdad4` |

기본 checkout의 브랜치를 바꾸지 말고 해당 worktree에서 작업한다.
최신 실험 코드 커밋은 `2faebd6`, 실행 안내까지 포함한 HEAD는 `6272cfd`다.
완료된 실험의 코드·결과표·학습곡선·청취 샘플을 각 실험 브랜치에 보존한다. 원격의 최신 상태와 커밋은 `git ls-remote --heads origin`으로 확인한다.
`runs/`는 Git 제외 대상이며 대용량 모델 체크포인트·실행 환경·캐시는 서버 로컬에 있다.
실험 패키지와 과거 소스는 protocol의 SHA256으로 고정돼 있다. 후속 변경은 별도 실험으로 만들어 재현성을 보존한다.

## 데이터와 평가 구성

- 입력: 16kHz mono, 1.5초(24,000 samples), 기존 `datasets/task2/fixed-1p5-v1` 사용. 재전처리하지 않았다.
- 총5화자, 각203개(명령200개+silence3개), 총1,015개. 명령10종+silence, 총11클래스.
- 개발 화자02~05의 leave-one-speaker-out 4-fold. 매 fold 나머지3명609개 학습, 해당1명203개 검증.
- test는 **spk01 강재훈**이다. 최근 실험에서는 test 음성을 읽거나 평가하지 않았다. validation 결과를 test 점수라고 표현하면 안 된다.
- 화자02 곽민채가 유일한 여성이라는 사용자 설명. 03 노정민, 04 이서준, 05 최환준.
- 02 검증 fold는 남성3명만 학습하므로 성별 차이가 가능한 가설이지만, 녹음 환경·개인차와 분리되지 않아 원인으로 확정할 수 없다.
- epoch 선택: 고정0dB draw0에서8종 소음별 macro-F1의 평균, 엄격한 최고점 갱신/동점은 이른 epoch.
- 최종 검증: 동일한8소음×3draw의0dB, 원본, 잡음 단독 입력. 최종 표는 화자별 결과의4-fold 평균이다.
- 3draw는 소음 추출 반복이며 학습 seed 반복이 아니다. 최신 protocol seed42, 다중 seed 유의성 검증은 안 했다.
- 0dB는 기존 혼합 구현 기준이다. 원본에 이미 있는 배경을 제거한 순수 발화 에너지 기준이라고 주장하면 안 된다.

## 현재 모델: 이전 대화의 잘못된 표에 주의

현재 실제 모델은 SpeechBrain `asr-conformer-transformerlm-ksponspeech`의 한국어 음성인식 사전학습 인코더를 사용한다.
원본 revision은 `ecb74648052c89618cd1ce53ccbc909d1141f164`다.

`1.5초 → 80-bin log-mel(FFT400, 25ms/10ms) → 사전학습 고정 전역 정규화 → CNN3블록 → 1280→256 투영 → Conformer12블록 → 시간축 평균 → 256→11 분류층`

- CNN은 시간축151→76→38, 주파수80→40→20, 채널64. 마지막 CNN 블록은 residual 포함.
- Conformer: hidden256, 4heads, FFN2048, convolution kernel31, 상대 위치 attention, 양방향.
- 블록: FFN half residual → MHSA residual → Conv residual → FFN half residual → LayerNorm.
- FFN 활성화는 실제 구현상 Swish/SiLU. 파라미터 대부분은 FFN에 있다.
- 전체 **32,122,699개** 파라미터, 전체 인코더 학습. ASR decoder/LM/CTC는 분류기에 포함하지 않는다.
- 과거 붙여 넣은 'Conv2개+Transformer6블록,512→11,약2,060만' 표는 현재 모델과 다르다.

## 직전 실험에서 선택한 기준 조건

`/root/audio/.worktrees/overnight/reports/overnight-v1/summary.csv` 참고.

| 조건 | 0dB F1 |
|---|---:|
| 당시 noise baseline | 89.7492% |
| Dropout0.2 | 90.6651% |
| Dropout0.2 + 약한 포먼트(formant_B) | **91.2974%** |
| 기존 SpecAugment·속도·피치 조합(augment_A) | 90.6826% |
| augment_A + 포먼트(augment_AB) | 90.8539% |
| 보조특징 set3 | 91.3015% |
| 보조특징 set7 | 90.9084% |
| WD1e-3 | 89.6436% |

set3의 개선은 약0.004%p라 채택하지 않았다. DFN3 후처리도 선택된 포먼트 모델에서 F1이 약90.41%로 낮아져 채택하지 않았다.
따라서 이번 기준은 **기존 소음+약한 포먼트+Dropout0.2, WD1e-4, 보조없음·잡음제거없음**이다.
직전과 최신 보고서의 `baseline` 이름은 다른 조건을 가리킨다. 최신 baseline은 직전의 `formant_B`다.

공통 학습 조건:

- 소음 혼합 확률0.5, SNR uniform[-5,20]dB, 기존6훈련 소음 원본 유지. 포먼트는 독립 확률0.5.
- 약한 포먼트: 파일별2변형, loguniform[1/1.1,1.1]. 기존 전체파형 변환 방식을 유지.
- Dropout0.2(frontend0.1), WD1e-4, batch16, 609draw/epoch(39updates).
- encoder LR1e-5 / head5e-4, 전체 인코더 학습. 새 freezing/LR 분리 비교를 추가하지 않았다.
- 같은 noise epoch10 LAST의 모델·AdamW·Torch/CUDA·샘플러 RNG에서11~100epoch(90epoch 추가), 조기 종료 없음.
- 기준은 기존 결과 재사용하되 기존 예측과 학습 draw의 일치를 사전 검증했다.
- 최신 실험에서 SpecAugment/속도/보조/필터를 추가하지 않았다.

## 최신 attention-pitch-v1: 구현

### attention 조건

- 고정 Silero VAD가 **실제 증강·소음 혼합된 입력**에서 발화 확률을 추정한다. 원본측 경계나 정답을 모델에 제공하지 않는다.
- 공식 가중치 revision `066409f3d9db45a30d21dfcbf9147826ffdd7052`.
- 32ms 창 확률을 CNN의40ms 위치에 정렬, 평활화 및 앞뒤80ms 확장.
- `g=.05+.95*p`, 최대 확률0.3 미만이면 전체 구간을 유지하는 fallback.
- 모든 Conformer 블록의 key logits에 `log(g_j)` 추가 + 최종 `sum(g*h)/sum(g)` 집계.
- 즉 **attention 억제와 가중 pooling을 함께 바꾼 조건**이다. 학습 결과의 개선을 attention 단독 효과로 분리할 수 없다.
- CNN/Conformer Conv는 유지하므로 배경 영향이 완전히 차단되는 구조는 아니다. 파라미터나 FLOPs를 줄이는 기법도 아니다.

### pitch 조건

- 기존 원본 자동VAD 경계를1.5초 좌표에 맞춰 발화 내부만 변형한다. 수동 정답 경계는 아니다.
- 구간 내부3~4개 무작위 앵커, 방향이 바뀌는 PCHIP 곡선, 구간 양끝 변화0.
- 파일당8변형, 최대 강도±1/1.5/2/2.5반음, 최대 기울기32반음/초, 선택 확률0.5.
- Praat pitch tier에 시간별 배율 적용 후 overlap-add 재합성. 내부 경계20ms crossfade.
- **새 피치 단계 직후 발화 바깥은 원본과 비트 단위로 동일**하다. 이후 기존 전체파형 포먼트 변형은 그대로 적용한다.
- 적용 순서: 새 국소 피치 → 기존 포먼트 → 소음 혼합. 추가 소음에는 피치를 적용하지 않는다.
- 발화와 겹친 원본 녹음 배경은 분리하지 못한다. `spk05_057.wav` 1개는 Praat 유성점 미검출로 피치 변형 없이 유지됐으며 기록돼 있다.
- 합성 모음의 실제 F0 추적 검사 통과. 실험 전 후보였던 RubberBand 방식은 곡선 추종 문제로 교체했으며 **실제 학습은 Praat**다.

### 경량화

12개 층 중 균등한 깊이의8개/6개를 남기고, 남은 파라미터와 AdamW 상태를 보존했다.
같은 epoch10 출발과100epoch 예산, 증류 없음. 아래 채택 기준이 baseline을 유지했으므로 **baseline 조건을 경량화했다. 관측 최고 attention을 경량화한 결과가 아니다.**

## 최신 최종 결과: 전체 완료

2026-11-13 14:51~17:13 KST, 약2시간21분25초. 신규20fold 모두100epoch 완료, optional skip/실패 없음.

| 조건 | 파라미터 | 0dB F1 | 0dB 정확도 | 기준 대비 F1 | 최저화자(02) F1 |
|---|---:|---:|---:|---:|---:|
| baseline | 32,122,699 | 91.2974% | 91.8719% | — | 81.9513% |
| attention | 32,122,699 | **91.6418%** | **92.0567%** | **+0.3444%p** | **84.0838%** |
| pitch | 32,122,699 | 90.8364% | 91.3588% | −0.4610%p | 81.0711% |
| attention_pitch | 32,122,699 | 91.3023% | 91.6256% | +0.0050%p | 83.0119% |
| compact8 | 21,567,307 | 88.9261% | 89.5782% | −2.3712%p | 75.4566% |
| compact6 | 16,289,611 | 85.3750% | 85.9093% | −5.9224%p | 67.8166% |

attention의 화자별 F1 변화: 02 **+2.1325%p**, 03 −0.1842%p, 04 −0.3233%p, 05 −0.2473%p.
원본 F1: baseline/attention/결합99.7726%, pitch99.6589%, compact8/6 약99.31%.
잡음단독 false-positive rate: baseline1.09375%, attention1.5625%, pitch2.34375%, 결합0.9375%, compact8 5%, compact6 3.90625%.

**관측 최고는 attention, 사전 규칙에 따른 공식 채택은 baseline**이다.
채택 규칙은 기준 대비 평균 F1 +0.5%p 이상이고 개별 화자 하락1%p 이내다. attention은 개선 폭0.3444%p라 통과하지 못했다.
경량화 허용은 선택12블록 대비 평균 F1 하락0.5%p/개별1%p/원본 정확도 평균0.5%p 이내인데 모두 실패했다.

해석: 02에서 마스킹의 가능성이 보이지만 나머지 화자의 소폭 하락으로 전체 이득은 작다. 이번 피치 정책은 개선하지 못했다.
일반적인 모든 피치 증강/모든 경량 모델이 실패한다고 일반화하면 안 된다. 다중 seed, 증류 등은 아직 비교하지 않았다.
발화 전후 배경에 불필요한 attention이 생긴다는 사용자 가설은 아직 인과적으로 입증한 것이 아니다.

## 결과 확인 및 다음 채팅 작업 위치

아래 경로는 모두 `/root/audio/.worktrees/attention-pitch/` 기준이다.

- 전체 요약: `reports/attention-pitch-v1/README.md`
- 평균 및 화자별: `reports/attention-pitch-v1/summary.csv`, `fold_results.csv`
- 학습곡선: `reports/attention-pitch-v1/speaker_curves.png`, `history.csv`
- 선택: `reports/attention-pitch-v1/selection.json`, `compression_selection.json`
- 고정 조건: `reports/attention-pitch-v1/protocol.json`
- 사전 검사: `reports/attention-pitch-v1/preflight.json`, `pitch_acoustic_check.json`
- 완료 후 검증: `reports/attention-pitch-v1/independent_audit.json`
- 진단: `reports/attention-pitch-v1/diagnostics.csv` (기존 best 모델에 pool-only/attention+pool 적용, 고정draw0만 평가. 최종3draw 결과와 혼동 금지)
- 원본 자동VAD 경계를 평가에 주는 `source_region_*` 진단은 배포 입력으로 사용할 수 없고 조건 선택에도 쓰지 않았다.
- 새 피치 청취: `reports/attention-pitch-v1/listening/README.md`, `{spk02,spk03}_comparison.wav`, `pitch_0..7.wav` 등
- 완료 증명: `runs/attention-pitch-v1/complete.json`, 실행 로그 `execution.log`
- 새 후보 체크포인트: `runs/attention-pitch-v1/candidates/{attention,pitch,attention_pitch,compact8,compact6}/{spk02,spk03,spk04,spk05}/best.pt`
- 후보별 예측·학습추적은 같은 후보 폴더와 보고서의 `folds/` 아래 보존돼 있다. 완료 검증 후 LAST optimizer만 정리됐으므로 LAST가 없다고 미완료로 오해하지 않는다.
- 소스: `experiments/attention_pitch/{model,vad,waveforms,data,run,checks,report}.py`

완료 후 검증 기록: epoch 예측3,288,600개, 최종 예측104,700개 재계산, 학습 draw1,096,200개 확인.
소스 hash와 짝지은 소음 source/crop/SNR 보존 확인. test_audio_read=false, test_evaluated=false.
앞선 채팅에서는 화자별 CSV로 평균과 변화량을 다시 계산해 summary.csv와 일치함도 확인했다.

실행 환경: `/root/audio/.worktrees/pretrain/runs/denoise-env/bin/python` (torch2.6.0+cu124, SpeechBrain1.0.3, parselmouth0.4.6).
당시 GPU는 RTX3090 24GB. 새 실행 전 실제 GPU/여유 디스크 확인. Python은 `PYTHONUTF8=1` 또는 명시적 UTF-8 파일 I/O 사용.
의존 데이터/과거 runs 일부는 다른 worktree로 symlink돼 있으므로 과거 worktree와 runs를 삭제하지 않는다.

## 이어갈 때

사용자가 다음 실험을 아직 지정하지 않았다. 현재는 결과 정리와 인계까지만 요청했다.
필요하면 02의 마스킹 개선이 어떤 소음/오류에서 생겼는지, VAD gate 및 pooling/attention 기여를 분석하는 방향을 논의할 수 있다.
이는 제안이며 합의된 추가 학습 계획은 아니다. 기존 완료 실험을 자동으로 재실행하지 않는다.
