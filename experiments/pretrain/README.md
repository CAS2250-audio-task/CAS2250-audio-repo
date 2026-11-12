# ref 기준 스펙트로그램 음성 사전학습

`pretrain`은 원격 `reference/ai-baseline`의 `6d208bc`에서 시작했다.
기존 `jaehun` 실험은 `cb9dd65`에 별도로 보존했다.

ref의 `src/`, `configs/task2_real.json`, `fixed-1p5-v1`은 변경하지 않는다.
16kHz·1.5초·1채널128-bin log-mel, ResNet18과512→11 분류층을 그대로 쓴다.
Train609(spk02/03/04), validation203(spk05), 10epoch, 무증강,
AdamW LR5e-4/WD1e-4, cosine 스케줄도 ref의 원래 실행기를 재사용한다.

## 사전학습 가중치

- 원본: [CLAR 공식 저장소](https://github.com/haideraltahan/CLAR).
- 공개 가중치: `sp_2d_100_fd_tm/sc09/epoch=1000.ckpt`.
- 사전학습 과제: Speech Commands SC09의 영어 숫자 발화 분류.
- 원래 입력: STFT magnitude(dB), STFT phase, log-mel의3채널.
- 전이: 첫 conv의 log-mel 채널(index2)을 선택하고 본체120개 텐서를 모두 복사한다.
- 원래 분류층은 제외한다. ref의512→11층은 같은 시드의 ImageNet 대조군과 동일하게 초기화한다.
- SHA-256: `ac36f2f4d6f25dfe792a2ea4699cc046630903decc5c948a700554741bbc5b7f`.

스펙트로그램 기반 음성 분류 사전학습이며 한국어 받아쓰기 ASR 모델과는 과제가 다르다.
영어 숫자·3채널에서 한국어 명령어·1채널로 옮긴 차이는 결과 해석에 포함한다.
[CLAR MIT 라이선스](CLAR_LICENSE.txt)와 보고서의 공개 모델 출처를 함께 보존한다.

## 실행

현재 서버에서는 기존 환경을 사용하며 새 패키지를 설치하지 않았다.

```bash
cd /root/audio/.worktrees/pretrain
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  /root/audio/.venv/bin/python -m experiments.pretrain.run
/root/audio/.venv/bin/python -m experiments.report_speech_pretrain
```

체크포인트가 없으면 공개 원본을 다운로드하고 위의 해시를 확인한다.
구형 Lightning 메타데이터는 무동작 딕셔너리 대체형으로 읽으며 `weights_only=True`를 유지한다.
외부 Lightning 코드나 저장된 함수는 실행하지 않는다.

같은3시드(42/43/44)의 ImageNet/음성 초기화를 비교한다. 완료 실행은 재사용한다.
코드 해시가 달라졌으면 새 출력 경로를 쓴다. 가중치·캐시는 `runs/`, 공유 보고서는 `reports/`에 둔다.
test spk01은 학습·모델 선택·추론에 사용하지 않는다.

[완료 결과](../../reports/speech-pretrain-ref-v1/README.md)
