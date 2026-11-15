# 공통 CNN 3종 비교 및 지도대조학습 인수인계

이번 서버는 외부 사전학습 없이 `ce` 조건의 공통 CNN 3종을 학습한다. 검출기와 발화확률을 사용하지 않고 전체 1.5초를 입력한다. 다른 서버에서는 같은 캐시와 초기화로 `paired_ce` 대조군과 `paired_supcon`을 비교할 수 있다.

## 동일 조건

| 집계 방식 | 추론 파라미터 | 공통 CNN+투영 |
|---|---:|---:|
| 평균 pooling | 77,835 | 77,472 |
| ST-inspired pooled-query, 1 head | 78,859 | 77,472 |
| 작은 self-attention + CLS, 1 block / d32 / heads2 / FFN64 | 86,315 | 77,472 |

논문 원 모델을 재현한 것이 아니라 동일 CNN 뒤의 집계 방식을 비교하는 자체 구성이다. 셀프 어텐션에는 Transformer 블록 전체와 위치 정보가 추가된다. 세 모델은 CNN과 분류기의 초기 텐서가 같고, 이후 독립적으로 학습한다.

- 개발 화자02~05의 812개만 사용한다. spk01 음원은 읽지 않는다.
- spk02와 spk03을 각각 홀드아웃하는 2개 독립 분할: 매 분할 학습609 / 검증203. 두 화자를 동시에 제외하는 분할이 아니다.
- seed42, 100epoch, batch32, AdamW lr1e-3 → cosine → 1e-5, weight_decay1e-4, gradient_clip_norm5.
- 64채널 log10-mel: 16kHz, FFT512, window400, hop160, 20~7800Hz, center=True, 151frames. 주파수별 평균·표준편차는 각 분할의 학습 원본만으로 산출한다.
- 약한 포먼트: p0.5, loguniform[1/1.1,1.1], 원본별 고정 2변형. pitch median 유지. 캐시의 변형 파형을 그대로 공유한다.
- 소음은 포먼트 변환 뒤 혼합한다. SNR U[-5,20]dB. 전체 1.5초의 명령 파워를 쓰며 silence는 학습 명령의 중앙 RMS를 기준으로 한다. 피크 초과 시 신호와 소음을 함께 줄인다.
- `ce`는 소음 p0.5. `paired_ce`와 `paired_supcon`은 추가 소음 p1.0으로 같은 원본/혼합 쌍을 만든다. paired 두 조건끼리 대조해야 한다.
- 같은 objective 내 세 모델은 매 epoch 같은 순서·증강 파형·특징을 받는다. 서로 다른 서버도 epoch·분할·원본 인덱스를 기반으로 같은 난수를 생성한다.
- 기존 평가 소음8종의 고정0dB draw0에서 소음별 macroF1 평균이 가장 높은 epoch를 모델별로 선택한다. 최종 best/last는 원본 및 0dB 3draw로 평가하고 best의 잡음 단독 오반응률을 기록한다.

## 다른 서버에서 복원

Python3.10, CUDA 사용 가능한 환경에서 다음을 실행한다. 이번 실행 환경은 RTX3090 / torch2.3.1+cu121이며 자세한 버전은 protocol.json과 requirements.txt에 고정한다.

```bash
git clone --branch compact-pooling-v1 https://github.com/CAS2250-audio-task/CAS2250-audio-repo.git
cd CAS2250-audio-repo
python -m pip install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r experiments/compact_pooling/requirements.txt
```

릴리스 `compact-pooling-v1-assets`의 캐시 묶음과 체크섬 파일을 받는다. 릴리스 업로드가 완료돼야 아래 다운로드 명령을 사용할 수 있다.

```bash
gh release download compact-pooling-v1-assets --repo CAS2250-audio-task/CAS2250-audio-repo --pattern 'compact-pooling-v1-assets.tar.gz*'
sha256sum -c compact-pooling-v1-assets.tar.gz.sha256
python -m experiments.compact_pooling.bundle restore compact-pooling-v1-assets.tar.gz
```

묶음에는 현재 실험의 포먼트 캐시, 학습/평가용 준비된 소음 파형14종, 개발812개 음원, 메타데이터·전처리 레시피·설정, 파일별 SHA256을 포함한다. 결과를 포함해 묶으면 프로토콜·609draw/epoch 증강 추적·체크포인트·선택 모델 예측도 함께 들어간다. 대용량 원본 소음 ZIP/WAV는 실행에 필요하지 않아 중복 포장하지 않는다. 원 출처와 원본 체크섬은 sources.json에 보존한다.

이 묶음은 **이번 실행에서 생성한 캐시를 두 서버에 동일하게 제공**한다. 이전 서버에서 소실된 증강 캐시를 바이트 단위로 복구했다고 의미하지 않는다. 준비 WAV의 파일 헤더·실행 라이브러리 차이가 있을 수 있으므로 다른 서버는 다시 생성하지 말고 업로드된 묶음을 복원한다.

모델 파일은 CNN·분류기만으로 추론할 수 있다. 입력 정규화의 mean/std는 best 체크포인트 안에 있고, frontend 설정·레이블 순서는 protocol.json을 따른다.

## 실행 명령

현재 서버의 3종 비교:

```bash
python -u -m experiments.compact_pooling.run
```

다른 서버의 지도대조학습 비교:

```bash
python -u -m experiments.compact_pooling.run --objective paired_ce --output runs/compact-pooling-paired-ce
python -u -m experiments.compact_pooling.run --objective paired_supcon --output runs/compact-pooling-paired-supcon --supcon-weight 0.1 --temperature 0.1
```

각 명령은 3모델 × 2분할을 실행한다. 먼저 한 모델만 비교하려면 **두 명령 모두** 동일한 `--models cnn_mean` 또는 다른 모델 이름을 지정한다. 원래 ce 결과 경로에 paired 실험을 덮어쓰지 못하게 막아뒀다.

지도대조학습에서는 같은 포먼트 변환을 적용한 원본과 추가 소음본을 사용한다. 원본에 이미 배경 소음이 있을 수 있어 깨끗한 정답 파형이라고 가정하지 않는다. 두 입력 모두에 분류 손실을 적용하며, 같은 명령어는 다른 화자·다른 녹음도 positive로 취급한다. silence는 분류 손실에만 사용한다. SupCon은 `L = CE + 0.1*SupCon`, temperature0.1이며, 32→64→32 투영부4,192개 파라미터는 학습에만 사용하고 배포에서는 제거한다.

원본·추가 잡음 쌍의 입력 수와 SNR 정책이 같도록 `paired_ce`와 비교한다. 기본 `ce`와의 차이를 전부 대조 손실 효과라고 해석하면 안 된다. 다른 화자의 같은 명령어를 positive로 구성하더라도 검증 화자를 학습에 포함하지 않는다.

## 기록, 재개, 검증

- `reports/<실험명>/protocol.json`: 소스·음원·소음 해시, 분할, objective, 모든 학습 조건.
- `runs/<실험명>/<화자>/history.json`: 모든 epoch의 학습/원본/0dB 기록.
- `augmentation_NNN.json`: 세 모델이 실제로 공유한 draw 순서·포먼트 선택·소음 ID·crop·SNR.
- `*_best.pt`, `last.pt`: 선택 모델과 10epoch 간격의 전체 모델·optimizer 상태. 같은 명령을 재실행하면 마지막 저장 epoch부터 재개한다. protocol이 달라지면 중단한다.
- `*_best_predictions.csv.gz`, `*_final_metrics.json`: 최종 예측과 소음별 지표.
- `reports/<실험명>/README.md`, `summary.json`: 최종 결과.

실제 개발 음원 전처리·세 모델 역전파·공통 CNN 초기값·증강 난수 재현성·화자 분리를 확인했다. SupCon은 양성/음성 마스크와 silence 처리, 세 모델 역전파 및 별도2epoch 전체 저장/평가 smoke 실행을 확인했다. 이 smoke 결과는 성능 결과가 아니다.

한 seed·개발2분할의 선별 결과다. 체크포인트 선택에 쓰인 draw0가 최종3draw에도 포함된다. 기존 Conformer의4분할·사전학습 결과와 직접적인 개선 수치로 비교할 수 없다.

## 캐시를 새로 준비해야 할 때

동일 실험을 복제할 때는 위의 릴리스 묶음 복원을 우선한다. 새 캐시를 만드는 명령은 아래와 같다.

```bash
python -u -m experiments.compact_pooling.prepare_formants
python -u -m experiments.compact_pooling.prepare_sources
python -m experiments.compact_pooling.bundle pack
```

원본 소음 다운로드는 publisher MD5를 검증하고, 생성된 파일의 SHA256을 기록한다. Praat 변환은 전역 상태 충돌을 피하기 위해 별도 프로세스에서 실행한다. `bundle pack`은 학습 완료 결과까지 묶으며, 캐시만 묶을 때는 `--cache-only`를 지정한다.
