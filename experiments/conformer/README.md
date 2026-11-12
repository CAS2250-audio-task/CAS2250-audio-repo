# 한국어 Conformer: ref 기준 전이

공개 `speechbrain/asr-conformer-transformerlm-ksponspeech`의 고정 버전
`ecb74648052c89618cd1ce53ccbc909d1141f164`를 사용한다. KsponSpeech 965.2시간
한국어 ASR의 CNN 앞단·12층 Conformer 인코더와 원래 global 정규화 통계를 가져온다.
모든 인코더 텐서의 이름·형태를 확인하고 strict load한다. ASR 디코더·언어 모델은
사용하지 않고 평균 풀링과 새 `256→11` 분류층을 붙인다.

완료 결과: 고정 / 미세조정의3시드 평균 검증 macro-F1은 **98.48% / 99.85%**다.
이전 ImageNet 대조군은89.26%다. [전체 결과·곡선](../../reports/conformer-ref-v1/README.md).
고정은2,827개만 학습하고 전체 모델은32,122,699개다. 총6회·60epoch, 전체 테스트36개 통과.

- ref fixed-1p5-v1: train609(spk02/03/04), validation203(spk05), test203(spk01).
- 원래 16kHz·1.5초 WAV를 그대로 읽으며 추가 오디오 패딩·크롭·증강은 없다.
- 원 모델의 80-bin Fbank, FFT400, 25ms 창, 10ms hop, f_min0과 pretrained global mean/std.
- **frozen**: CNN/인코더 eval+동결; 분류층만 LR5e-4로 학습.
- **finetune**: CNN/인코더 LR1e-5, 분류층 LR5e-4. 원 모델 Dropout0.1 활성.
- 공통: 3시드42/43/44, 10epoch, batch32, AdamW WD1e-4, cosine,
  원 ref와 같은 역빈도 weighted replacement sampler, epoch당609개 추출.
- 각 실행은 검증 macro-F1 최고 epoch를 선택한다. 테스트 화자 추론은 하지 않는다.
- 같은 시드 두 조건의 초기 분류층·인코더는 동일하다.
- 기존 ImageNet 결과는 `reports/speech-pretrain-ref-v1`의 동시 환경 대조군을 사용한다.
  구조·특징·인코더 학습률까지 바뀌므로 가중치 초기화만의 통제 실험은 아니다.

기존 `.venv`를 변경하지 않고 `runs/conformer-env`에 별도 환경을 만들었다.
그 환경의 `audio_base_venv.pth`는 `/root/audio/.venv/lib/python3.10/site-packages`를
참조하며 위 requirements의 추가 패키지만 별도로 설치했다. torch/torchaudio는 기존2.6.0+cu124다.

```bash
cd /root/audio/.worktrees/pretrain
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  runs/conformer-env/bin/python -u -m experiments.conformer.run
runs/conformer-env/bin/python -m experiments.report_conformer
```

`runs/conformer-ref-v1/protocol.json`은 실행 전 조건·소스 해시를 고정한다.
`last.pt`에 optimizer/scheduler/RNG/sampler 상태를 매 epoch 저장하므로 같은 명령으로 재개할 수 있다.
완료된 실행은 건너뛴다. 보고서 위치는 `reports/conformer-ref-v1`이다.
