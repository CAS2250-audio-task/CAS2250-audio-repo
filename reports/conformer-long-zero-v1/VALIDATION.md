# 실행·검증·재개

- 전체 회귀 검사: `runs/denoise-env/bin/python -m pytest -q` — **58 passed**.
- `preflight.json`: 기존 공식 원본/0dB 예측3,654개 재현. batch16과 마지막batch1의손실/역전파유한값확인.
- `initial_pairing_audit.json`: 초기완료pilot의같은조건·epoch에서배치/LR이달라도같은609입력순서와증강파형해시인지확인했다. 완료시전체를다시검사한다.
- `environment.json`: 실제Python/Torch/torchaudio/SpeechBrain/NumPy/DFN환경과GPU기록.
- `protocol.json`: 원래코드·sourceepoch10checkpoint·검증원본/DFN캐시·개발WAV해시고정. 훈련코드는실행중수정하지않는다.

전체48pilot·16장기분기 학습과 핵심 독립 검사가 완료됐다. `independent_audit.json`과 `runs/conformer-long-zero-v1/complete.json`이 존재한다. 추가 비교표·문서·해시 검사 완료는 `postprocess_complete.json`으로 확인한다.

독립 검사에서 1,243에포크의 검증 예측2,270,961개, 최종 예측335,040개의 지표를 재계산했다. 최고본58,464개 예측을 재추론해 전부 일치했다. 훈련727,755draw와 각checkpoint의 실제AdamW 업데이트 수·RNG상태를 확인했다. 재개 환경의 전체 회귀 검사도58개 통과했다(`regression_tests.json`).

중단/재부팅후먼저`runs/conformer-long-zero-v1/process.json`의PID와`progress.json`, `execution.log`를확인한다. 살아있는driver가있으면중복실행하지않는다. 없으면pretrain작업폴더에서아래명령으로재개한다.

```bash
runs/denoise-env/bin/python -u -m experiments.conformer_long_zero.run
```

각분기의`last.pt`는모델·optimizer·torch/CUDA RNG·sampler상태·완료epoch/history를원자적으로저장한다. 미완료epoch는마지막완료epoch에서다시계산한다. 완료pilot는재사용하며학습률선택은고정된pilot_history로재현한다. 장기종료후최종평가중끊겼으면이미정지한모델을더학습시키지않고평가를재개한다. 파일잠금으로동시driver를막는다.

중간그래프수동갱신:

```bash
runs/denoise-env/bin/python -m experiments.report_conformer_long_zero
```

전체학습후driver가독립검사를자동실행한다. 수동재검사:

```bash
runs/denoise-env/bin/python -m experiments.check_conformer_long_zero
```

독립검사는epoch별검증F1/손실재계산,배치/LR간입력짝검사,LR선택재계산,중단판정과checkpoint선택재계산,최종전체3draw지표재계산,최고checkpoint의원본/0dB off·DFN3예측재추론,원래코드/가중치/데이터해시보존을검사한다.

원본검증과증강학습은입력분포가다르므로손실절댓값의크기차이만으로과적합을판단하지않는다. 고정된0dB검증의시간추세와훈련손실의시간추세를본다. epoch10행의훈련손실은기존학습값이고LR열은이후분기의설정값이다. 새LR을적용하는첫epoch는11이다.

개발소음/화자를LR·epoch선택에사용하며전체3draw도독립test가아니다. test01오디오는이실험에서읽거나추론하지않는다.

학습/독립검사가끝나면추가분석과최종보고서검사를아래순서로실행한다.

```bash
runs/denoise-env/bin/python -m experiments.analyze_conformer_long_zero
runs/denoise-env/bin/python -m experiments.report_conformer_long_zero
runs/denoise-env/bin/python -m experiments.check_conformer_long_zero_analysis
```

추가검사는이전공식0dB집계와의일치,학습driver최종집계와의일치,선택draw0/추가draw1~2의전체평균복원,동일epoch비교,문서링크와산출물해시를확인한다. `postprocess_complete.json`이최종분석완료표시다. 이검사뒤보고서를다시생성하면산출물해시목록도새로생성해야한다.
