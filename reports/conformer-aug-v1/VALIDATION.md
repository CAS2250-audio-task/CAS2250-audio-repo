# Conformer 증강 검증 기록

원래 ref 데이터와 모델 구조를 유지하고 새12회·120epoch를 학습했다. 기존 무증강4개는 재사용했다.
모든 새 실행에서 원래 ASR 인코더와 같은 seed42 분류층으로 초기화됐음을 baseline 초기 해시와 대조했다.

- 전체 테스트48개 통과. 새5개는 피치 유지 속도 변형·검수 구간 보호·native 무음값 마스킹·훈련 소음 재현·평가 시 마스킹 해제를 확인한다.
- 전체325,296개 예측을 기록했다. 기존 baseline81,324개와 새 모델243,972개다.
- 기존 검증 혼합80,512개(음성+잡음77,952·잡음단독2,560)를 다시 생성해 기존 파형 SHA256과 모두 일치했다.
- 독립 검사에서2,064개 조건의 정확도와11클래스 macro-F1을 저장 예측으로 다시 계산했다.
- 120개 epoch의 검증 예측24,360개를 재계산해history와 일치했고, 최고epoch가 원본validation F1의 첫 최댓값인지 확인했다.
- 새12개 최고checkpoint에서 원본train/val9,744개 예측을 독립 재추론해 일치했다.
- 화자·소음·SNR 조합마다 명령어/silence를 하나씩 골라256개 혼합을 독립 복원했다. 새 모델768개 잡음 예측이 저장 결과와 일치했다.
- 훈련draw73,080개가 해당fold의train 경로 안에만 존재하며 검증·test화자가 들어가지 않았음을 확인했다. 조건 간 같은epoch/draw의sample 순서·속도선택·소음선택/원본/구간/SNR도 짝지어 일치했다.
- 훈련소음6개와 검증소음8개는 원본파일명·원본SHA256·준비WAV SHA256 교집합이0이다.
- 개발원본812개·기존baseline4개·훈련/평가소음·동결코드의 해시와 사전학습 정규화 통계가 유지됐다.
- silence기준은 각fold의훈련명령어600개RMS중앙값으로 독립 재계산해 일치했다.
- 속도은행 명령어변형1,600개 중17개는 검수구간 보존을 위해 건너뛰었고 가속 추가0패딩은 최대135.8ms다.
- 이번 실험에서testspk01은 학습·검증·잡음평가에 사용하지 않았다. 프로젝트 이전 실험의test평가 이력은 별도로 존재한다.

환경은 기존Conformer 전용venv, RTX3090, float32·TF32 on이다. 네개발화자·한학습시드의선별 결과이며 통계적 우월성이나 최종test성능으로 해석하지 않는다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 runs/conformer-env/bin/python -m pytest -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 runs/conformer-env/bin/python -m experiments.check_conformer_aug
```

[독립 검사](independent_audit.json) · [평가 감사](evaluation_audit.json) · [고정 프로토콜](protocol.json)
