# 화자03 · 100→1000epoch 연장

기존 `compact-pooling-v1`의 **화자03 분할**만 대상으로 평균 풀링·풀드 어텐션·셀프 어텐션 세 모델을 총1000epoch까지 이어 학습한다. 학습 화자는02·04·05이며, 검증은03이다. 원래100epoch 실험과 화자02 검증 실험은 보존한다.

```bash
python -u -m experiments.compact_pooling_long.run
```

먼저 기존 릴리스 `compact-pooling-v1-assets`를 복원해야 한다. 원본 데이터·포먼트 캐시·소음·정규화·증강 확률·SNR·batch32·weight decay·선택 기준을 그대로 사용한다.

- 초기화: 기존 `runs/compact-pooling-v1/spk03/last.pt`의100epoch LAST 모델과 AdamW 상태. 선택된 BEST 가중치에서 출발하지 않는다.
- 재현 확인: 기존100epoch 원본 정규화 평균/표준편차 및 모델의0dB draw0 F1을 재계산해 확인한다.
- 학습률:101epoch에서 `1e-4`로 재설정하고,1000epoch까지 하나의 cosine으로 `1e-5`까지 낮춘다. AdamW 모멘트는 유지한다. 기존100epoch의 cosine을 처음부터1000epoch로 늘려 다시 학습하는 조건과 다르다.
- 총 학습량: 모델별 추가900epoch,609개/epoch. 세 모델은 같은 증강 입력을 받고 독립적으로 업데이트된다. 증강 난수는 절대 epoch101..1000을 사용한다.
- 선택: 기존100epoch에서 고른 BEST도 후보에 포함하고,0dB draw0의 소음별 macroF1 평균이 엄격히 개선될 때만 교체한다.
- 저장:10epoch마다 전체 LAST 모델·optimizer;200/500/1000epoch의 전체 상태도 별도 보존한다. 모든 epoch의 손실·원본/0dB F1과 증강 draw를 기록한다.
- 최종 평가: 선택 BEST와1000epoch LAST를 원본 및0dB 3draw로 평가한다. BEST의 잡음 단독 오반응률도 기록한다.
- 실행 도중 중단됐다면 동일 명령을 재실행하면 마지막 저장 epoch부터 재개한다. 실행 조건이 바뀌면 프로토콜 검사에서 중단한다.

출력은 `runs/compact-pooling-spk03-1000-v1`와 `reports/compact-pooling-spk03-1000-v1`이다. one-seed·개발 화자1명 결과이며 독립 test 성능이 아니다. 원본 녹음에는 이미 배경 소음이 있을 수 있다.

기본 실행의 예상 학습·검증 시간은 RTX3090에서 약15~20분이며 저장과 최종 평가가 추가된다.
