# 잡음 제거 평가 검증

- 분류기: 소음 혼합 학습본의 기존 clean-best checkpoint 4개만 사용. 새 분류기/잡음 제거기 학습 없음.
- 평가 입력: 원본812, 동일0dB 혼합19,488, 잡음 단독640. spk02~05만 포함; test01 오디오 읽기/추론 없음.
- 기존 혼합20,128개의 SHA256 일치. CSV에 반올림된 reference power를 직접 쓰면 드물게 해시가 달라져, 원래와 같이 원본 파형 및 fold 훈련600개 명령어 RMS에서 정확히 재계산했다.
- 필터 없는20,940개의 예측이 이전 소음 혼합 학습본과 모두 일치한다.
- 총83,760개 출력 파형의 파일/개별 파형 해시와 유한값·길이·dtype를 확인했다.
- 세 필터는 입력 파형만 받는다. 정답/소음 종류/분리된 잡음/깨끗한 원본/SNR을 전달하지 않는다. 원본 파워는 기존 혼합 생성에만 쓰였다.
- 각 필터의 반복 호출은 bitwise 동일, 배치 순서를 뒤집어도 동일했다. batch1 대비 batch8의 RMS 차이는 DFN3 약0.0044%, DNS64 약0.0148%, NSNet2 0%였다. 작은 차이는 GPU 배치 수치 오차 범위다.
- NSNet2 배치 구현은 Microsoft 공식16kHz featurelib/ONNX/합성 경로와 파형이 정확히 일치했다. DFN3의 리샘플링도 공식 sinc_fast와 정확히 일치했다. [사전 검사](preflight.json)
- 독립 검사에서 528개 조건의 accuracy/macro-F1/잡음 오반응률을 다시 계산했다.
- 실제 평가 배치와 같은 경계에서816개 파형을 새로 필터링해 정확히 일치했다. 고정 분류기의1,840개 예측을 재추론해 모두 일치했다.
- 기존 데이터812개, 소음8개, checkpoint4개, 모델 코드/가중치/입력 캐시/이전 보고서의 해시가 고정 프로토콜과 일치한다. [독립 검사](independent_audit.json)
- 프로젝트 테스트 **51개 통과**. 처음 기본 pytest가 ignored vendor 저장소의 별도 연구 테스트까지 수집해 의존성 오류를 냈다. `pytest.ini`에 프로젝트의 `tests` 경로를 지정한 뒤 기본 명령으로 다시 통과했다. 외부 저장소의 훈련/데이터셋 테스트까지 통과했다는 의미는 아니다.
- 기존 Conformer/베이스 학습 환경은 변경하지 않았다. 별도 `runs/denoise-env`에4개 추론 의존성을 설치했다.
- 외부 잡음 제거기의 DNS 사전학습 전체 원본과 평가 잡음의 중복은 완전히 감사하지 못했다. 우리 분류기 훈련에서 미관측인 소음이라는 범위로 해석한다.

## 현재 작업 폴더에서 확인/재개

작업 폴더는 `/root/audio/.worktrees/pretrain`, 브랜치는 `pretrain`이다. 대형 가중치·필터 출력·입력 캐시·듣기 WAV는 ignored `runs/`에 있다. 보고서와 코드만 Git에 보존한다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 runs/denoise-env/bin/python -m experiments.check_denoise_eval
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 runs/denoise-env/bin/python -m pytest -q
```

중단한 평가를 재개할 때는 `runs/denoise-eval-v1/process.json`의 프로세스가 끝났는지 먼저 확인하고 다음 명령을 쓴다. 완료 fold는 재사용한다. 현재 모든 fold가 완료됐다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 runs/denoise-env/bin/python -u -m experiments.denoise_eval.run
```

외부 코드/가중치를 잃었을 때 `python -m experiments.prepare_denoise_assets`는 공식 revision과 전체 SHA256에 맞는 파일을 복구한다. 필요한 추가 의존성은 `experiments/denoise_eval_requirements.txt`, Conformer 의존성은 기존 환경을 따른다. 이 명령은 분류기 checkpoint·원본 데이터·평가 입력/출력 캐시를 복구하지 않는다.

`experiments.denoise_eval.prepare`와 `preflight`는 최초 protocol 생성용이다. 이번 고정 study에 다시 실행하면 생성 시간 등이 포함된 캐시/검사 파일 해시가 바뀔 수 있으므로, 이미 완료된 study에서는 위 독립 검사와 run 재개 명령만 사용한다. 입력 캐시를 새로 만들거나 필터 조건/동결 소스를 바꿀 때는 새 study 버전으로 진행한다.

## 지연과 듣기

- 지연: 다른 평가 작업이 끝난 후 batch1·워밍업20회·100회. 전체창의 CPU 파형→필터→Conformer→CPU 라벨 반환. 별도 필터 시간도 측정했다. 스트리밍 알고리즘 지연이나 녹음 시작부터의 총 지연은 아니다.
- 듣기: 이전부터 제공한 spk02_052 출발 파일의 traffic/restaurant/compressor 0dB draw0를 그대로 사용했다. 실제 평가된12개 필터 입출력과 예측/해시를 기록했다. 정답/오답을 보고 다시 고르지 않았다. 일부는 필터가 원래 정답을 오답으로 바꾼 사례다.
- 모든 WAV는 float32, 개별 피크/볼륨 정규화 없이 저장했다. 이어듣기 사이의0.4초0구간은 청취 편의용이며 모델 입력에는 없다. OGG는 미리듣기용 손실 압축본이다.
