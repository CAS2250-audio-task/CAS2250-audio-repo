# 사전학습 실험 인계

## 최신 완료: 높은 성능 점검

- 사용자:99.85%가 너무 높은 것 아닌지 점검 요청, 이후 재개 요청.
- 붙여준 Conv2개+Transformer6층·512→11·약2,060만 표는 Whisper-base 설명이며 실제 Conformer와 다름을 정정.
- 별도 `experiments/conformer_sanity.py`에서 입력 대조, 훈련 라벨만 섞는 고정 인코더 대조,
  개발 화자02–05의4-fold를seed42 한 개·각10epoch로 점검했다.
- Conformer는 기존 미세조정 조건 그대로, ImageNet은 원래 ref 학습 그대로. 양쪽matmul TF32 on.
- spk05 Conformer seed42는 같은 분할의 기존 실행 재사용. 다른3fold Conformer+4fold ResNet 신규.
- test spk01은 추론하지 않는다. 원본데이터·기존99.85% 보고서·동결 프로토콜은 보존한다.
- 실행 `runs/conformer-env/bin/python -u -m experiments.conformer_sanity`.
  로그 `runs/conformer-sanity-v1/execution.log`, 보고서 `reports/conformer-sanity-v1`.
- 새 점검 스크립트의 프로토콜은 실행 시 소스 해시를 고정한다. 실행 도중 수정하지 말 것.
- 총CV8회·80epoch(신규7회+기존spk05 Conformer1회 재사용), 훈련 라벨 대조1회·10epoch 완료.
- 검증spk02/03/04/05 F1 ImageNet:59.77787/83.33417/86.57055/91.83300%,
  Conformer:99.10090/98.63551/99.54517/100%. 평균80.37890/99.32040%, +18.94150%p.
- 최고epoch선택 대신 마지막epoch10으로도 평균F1 ImageNet78.72135/Conformer99.20369%.
- 기존seed44 입력 대조 F1: 정상99.54517%, 전체0음성0.26478%, 오디오 순서 교환10.43294%,
  정답 순서 교환10.43294%. 파일순서·batch32→17 변경 시 파일별 예측 동일.
- 훈련 라벨만 섞은 동결 인코더+분류층10epoch: 최고검증F1 9.86836%, 정확도12.31527%.
- 개발4명812개에서 화자 간 원본SHA·원본파일명·WAV/PCM 정확 중복0.
  검증통계로normalization학습 없음, 모델입력은파형만. 변형·근사중복과 사전학습원음전체 중복은 미확인.
- 독립 확인 `experiments/check_conformer_sanity.py`로8개CV checkpoint의1,624개 검증예측 재추론 일치.
  80epoch16,240개+라벨대조10epoch2,030개 지표 재계산. 전체테스트39개 통과.
- 환경·거리·잡음·기기·발화스타일 metadata가 없어 해당조건의 일반화는 이번 결과로 판단할 수 없다.
  unknown없는11클래스, 개발4명·seed42 한 개의 결과다. test spk01은 계속 미평가.
- 점검 결론: 명백한 누수·평가오류 징후가 없으며 다른개발화자에서도높은성능이재현됨.
  실제환경99.85%를보장하는결과는아니다. 원래99.85%는spk05·3시드평균이며CV99.32%와집계가다르다.
- 새보고서 `reports/conformer-sanity-v1/README.md`, `speaker_cv.csv`, `cv_checkpoint_replay.json`.
  학습프로세스종료, GPU유휴.

## 최신 완료: 한국어 Conformer

- 사용자 요청으로 `experiments/conformer`에서 한국어 Conformer를 ref 데이터로 실험했다.
- 동결 분류층 학습 / 인코더 LR1e-5 미세조정 ×3시드42/43/44, 각10epoch.
- 공개 가중치 버전·해시와 원래 정규화 통계를 고정했다. 추가 오디오 패딩 없이1.5초를 입력한다.
- 공개 모델 `speechbrain/asr-conformer-transformerlm-ksponspeech`, revision
  `ecb74648052c89618cd1ce53ccbc909d1141f164`, checkpoint SHA256
  `8e718dc29b403dfaa8d2604c43c3666be3fa99e958b77e3c6ff387e94d4a174c`.
- 전체32,122,699개 파라미터, 본체406개 텐서 strict load. 고정 조건은2,827개 분류층만 학습.
- 실행 환경 `runs/conformer-env/bin/python`; 추가 의존성은 기존 `.venv`와 분리했다.
  torch/torchaudio2.6.0+cu124, SpeechBrain1.0.3. 원래 환경에 새 패키지를 설치하지 않았다.
- 총6회·60epoch 완료. 고정/미세조정의 평균 검증F1 **98.47695/99.84839%**,
  정확도98.35796/99.83580%. 기존 ImageNet 대조군F1 89.25642% 대비 **+9.22053/+10.59197%p**.
- 고정 seed42/43/44: F1 98.62865/99.09034/97.71187%, best epoch7/6/6,
  정답200/201/198개. 미세조정: F1 100/100/99.54517%, best epoch6/9/6, 정답203/203/202개.
- 공통 최고epoch는 두 조건 모두9. 평균F1은 실행별 최고 평균과 동일하다.
- 고정/미세조정의 원본 학습F1은96.77366/99.46103%. 단일 검증 화자가 학습 화자보다 쉬울 수 있다.
- 새6개 최고checkpoint의 train/val4,872개를 재추론해 저장 예측·지표 일치 확인.
  이전 대조군까지 총24,360개epoch 예측에서 지표 재계산, 원본1,015개 해시·초기head일치·
  동결본체 보존·정규화 통계 보존 확인. 전체 테스트36개 통과.
- 해석: 모델 구조·크기·80-bin native Fbank·정규화·인코더 LR도 바뀐 시스템 비교다.
  한국어 하나만의 효과가 아니다. SpeechBrain은 CUDA matmul TF32를 켜며 기존 대조군의 기본값은off였다.
  검증spk05의203개 개발 결과이고 testspk01은 이번에도 추론하지 않았다.
- 로그 `runs/conformer-ref-v1/execution.log`; 학습 명령
  `runs/conformer-env/bin/python -m experiments.conformer.run`.
  매 epoch의 `last.pt`로 중단 재개 가능하며 완료된 실행은 건너뛴다.
- 보고서 `reports/conformer-ref-v1/README.md`, `learning_curves.png`, `checkpoint_replay.json`,
  `execution_audit.json`; 보고서 재생성 `runs/conformer-env/bin/python -m experiments.report_conformer`.
- 원래 ref의 `src/`, 기본 설정·데이터는 변경하지 않았다. 학습 프로세스 종료, GPU 유휴.
- 아래 내용은 완료된 이전 CLAR 실험이며 Conformer 결과와 구분한다.

2026-11-12 사용자 요청: 기존 실험 커밋 후 ref에서 새 브랜치 생성,
ref 기준의 스펙트로그램 음성 사전학습 활용.

- 기존 `jaehun`은 `cb9dd65`에184회 CV와 후속 진단·새 데이터 보존. 커밋 전55개 테스트 통과.
- 새 `pretrain`은 `origin/reference/ai-baseline`의 `6d208bc`에서 생성했다.
  작업 폴더 `/root/audio/.worktrees/pretrain`; `/root/audio`는 `jaehun`이다.
- 사용자 지시에 따라 real-context 데이터·4-fold CV·보조 특징을 가져오지 않았다.
  ref fixed-1p5-v1의609/203/203 분할, log-mel/ResNet18/10epoch/무증강을 유지했다.
- 파형 기반 Wav2Vec2는 실행·설치하지 않았다. 공개 CLAR의 Speech Commands SC09 음성 분류
  ResNet18을 사용했다. 원본 버전은 `3b5a7828744333f8915de4e64d87a544c8ce91fd`.
- 원래 STFT magnitude·STFT phase·log-mel의3채널에서 mel 채널2 가중치를 첫 conv로 가져오고
  본체120개 텐서를 모두 복사했다. ref512→11 head는 ImageNet과 같은 시드에서 동일하게 새로 초기화한다.
- 구현 `experiments/pretrain/weights.py`, `run.py`; ref `src/`, 기본 설정·데이터 변경 없음.
  기존 Python 환경 유지. 가중치는 pinned SHA를 확인하고 `weights_only=True`로 읽는다.
- ImageNet/음성 초기화 ×3시드, 총6회·60epoch 완료. 평균 최고 검증 F1은89.2564/85.2105%,
  음성쪽−4.0459%p. 평균 정확도88.3415/84.7291%, 학습 원본F1 100.00/99.5959%.
- 시드42/43/44 F1 ImageNet/음성:92.3778/84.9034,90.1452/85.2053,85.2463/85.5228%.
  공통 최고epoch는 둘 다9, 평균F1 88.4336/84.2525%다.
- 음성만 정답/ImageNet만 정답은 seed42 10/22,43 12/22,44 22/22개다.
  파일별 예측·WAV 링크는 보고서의 seedXX_files.md와CSV에 있다.
- ref의 원래 seed42 기록 F1 92.58%와 이번 재학습92.38%는 완전히 일치하지 않는다.
  비교 기준은 같은 환경에서 이번에 함께 학습한 ImageNet 대조군이다.
- 영어 숫자/3채널→한국어 명령어/1채널 적응 차이가 있어 음성 사전학습 일반의 무효 판정이 아니다.
  이번 전이 방식의 개선이 없어 ImageNet 기준을 유지한다. 추가 LR/입력채널 탐색은 하지 않았다.
- test spk01 추론 없음. 한 검증 화자203개의 개발 결과이며3시드는 같은 파일을 반복한다.
- ref 테스트+전이 검증32개 통과. 보고서 생성기로6개checkpoint·동일 초기head·데이터1,015개 해시·
  120개 본체텐서·60epoch의12,180개 예측 지표를 검증했다.
- 진입점 `reports/speech-pretrain-ref-v1/README.md`, `execution_audit.json`, `protocol.json`,
  `pretraining_source.json`. 로컬 실행/가중치는 `runs/speech-pretrain-ref-v1`, `runs/pretrain-research`.
