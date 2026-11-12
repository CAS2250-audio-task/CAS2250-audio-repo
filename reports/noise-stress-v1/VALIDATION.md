# 소음 평가 검증 기록

개발 화자02–05의 기존 최고 checkpoint 8개로 384개 혼합 조건을 평가했다.
각 조건의 원본·추출 위치·혼합 파형은 두 모델이 공유하며 새로운 학습은 하지 않았다.

- 원본 재추론: 1,624개 예측이 기존 4-fold의 저장 예측과 일치했다.
- 전체 예측: 162,648개. 원본 1,624·혼합 155,904·잡음 단독 5,120개다.
- 지표: 독립 검사에서 1,032개 조건의 정확도와 11클래스 macro-F1을 재계산해 일치했다.
- 혼합 기록: 80,512개. SNR 계산 오차 최댓값은 약 1.93e-15dB다.
- 구간 일치: 19,488개 파일·소음·반복 조합이 4개 SNR에서 같은 추출 위치와 잡음 파워를 사용했다.
- 독립 복원: 모든 화자·소음·SNR 조합에서 명령어와 silence 예시를 하나씩 골라 총 256개 파형을 원본부터 복원했다. 혼합 SHA256이 일치했고 두 모델의 512개 예측도 다시 일치했다.
- 입력 보존: 개발 WAV 812개와 기존 checkpoint 8개, 설정·메타데이터·동결 코드의 SHA256이 유지됐다.
- silence 기준: 각 fold의 훈련 키워드 600개 RMS 중앙값을 독립적으로 재계산해 기록과 일치했다.
- 소음 원본: 다운로드 시 배포 MD5를 확인하고, mono16k 준비 파일의 SHA256을 평가와 재검증에서 확인했다.
- 신호 검증: SNR 성분비·공통 감쇠·구간 재현·배치별 dB 상한 분리를 포함한 새 테스트 4개 및 전체 테스트 43개 통과.
- 이번 실험에서는 test spk01의 오디오를 읽거나 추론하지 않았다. 이전 프로젝트의 다른 모델 test 평가 이력은 별도로 존재한다.

실행 환경은 기존 Conformer 전용 환경과 RTX3090이며 양 모델 모두 float32 추론, TF32 on이다.
원본 8개와 한 학습 시드의 개발 평가다. 반복 구간과 fold는 독립 실험 표본으로 해석하지 않는다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 runs/conformer-env/bin/python -m pytest -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 runs/conformer-env/bin/python -m experiments.check_noise_stress
```

[집계 검사](aggregation_audit.json) · [독립 재추론](independent_replay.json) · [고정 프로토콜](protocol.json)
