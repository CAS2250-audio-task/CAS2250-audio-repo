# 사전학습 실험 인계

## 최신 완료: 소음 혼합 학습본에 잡음 제거기 3개 비교

- 사용자 요청: 0dB 개선을 위해 필터 3개를 소음 혼합 학습본에만 평가한다.
- 고정 분류기: `runs/conformer-aug-v1/folds/spk02~05/noise/best.pt` 4개, 추가 학습 없음.
- 필터: DeepFilterNet3, Meta DNS64 Demucs, Microsoft NSNet2 16k. 공개 가중치와 공식 코드 revision/SHA 고정.
- 폴더 `experiments/denoise_eval/`, 결과 `reports/denoise-eval-v1/`, 큰 캐시와 로그 `runs/denoise-eval-v1/`.
- 0dB 혼합19,488개, 원본812개, 0dB 기준 잡음 단독640개를 동일하게 비교. test01 제외.
- 필터 입력은 혼합 파형뿐. 깨끗한 원본/분리된 잡음/라벨/SNR을 필터에 전달하지 않는다.
- 16k·1.5초를 유지하고 DFN만48k 왕복 리샘플. 공식 지연 보정. 임의 피크 정규화/클리핑/원본 혼합 없음.
- 세 필터 재호출 bitwise 일치 및 순서 독립성 통과. batch/single RMS 차이 <0.02%, NSNet2는 공식16k 경로와 정확히 일치.
- 별도 `runs/denoise-env` 설치; 기존 학습환경 변경 없음. 외부 DNS 사전학습 잡음의 평가 원본 중복은 완전히 감사하지 못한 한계.
- `process.json` PID와 `execution.log` 확인 후 중복 실행하지 않는다. 중단 시 같은 환경으로
  `python -u -m experiments.denoise_eval.run`을 재개. protocol 고정 후 소스를 변경하면 새 study를 쓴다.
- 총83,760개 예측 완료. 원본/0dB F1 및 잡음 단독 오반응률:
  - 없음:99.2043/78.4748%, 오반응4.21875%.
  - DFN3:95.6769/83.7984%, 오반응12.8125%.
  - DNS64:98.7530/83.2600%, 오반응38.125%.
  - NSNet2:97.6081/81.8614%, 오반응7.8125%.
- 0dB는 모두 개선됐지만 원본 손상/잡음 오반응 증가가 있어 기본 경로에 자동 채택하지 않았다. 특히 DNS64는 정답률 개선과 높은 오반응이 함께 나타남.
- DFN3는 압축기60.59→75.09%, 식당48.11→59.21%로 개선. 구내식당71.12→71.32%로 거의 변화 없음.
- 독립 검사:528개 조건 지표 재계산, 출력83,760개 SHA 확인, 파형816개 재필터링 정확히 일치, 분류1,840개 재추론 일치. 기존 분류기/데이터/소음/원래 보고서 해시 보존.
- 프로젝트51개 테스트 통과. vendor 외부 테스트까지 수집되던 문제는 pytest.ini testpaths=tests로 해결했다.
- batch1 필터만/전체 중앙값: 없음0.04/15.49ms,DFN3 15.00/30.59ms,DNS64 7.07/22.12ms,NSNet2 17.39/33.90ms. RTX3090+CPU2threads;100회. 창 확보/로딩/WAVdecode/통신 제외, 비스트리밍.
- `runs/denoise-eval-v1/listening/README.md`: 이전 출발/spk02_052를 같은3종0dB로 처리한 실제 입력 WAV/이어듣기. 오류 선별 없이 고정 예시 유지, 일부 필터 후 정답→오답도 있음.
- 최종 문서 `reports/denoise-eval-v1/README.md`, 검증/재개 방법 `VALIDATION.md`, 독립 검사 `experiments.check_denoise_eval`.
- 외부 코드/가중치 복구 `experiments.prepare_denoise_assets`, 별도 환경 의존성 `experiments/denoise_eval_requirements.txt`. 새study가 아니면 prepare/preflight를 다시 실행하지 않는다(시간 포함 파일 해시 변경).

## 추가 측정: 현재 Conformer latency

- 사용자 요청으로 실제 소음 혼합 학습 모델의 batch1·1.5초 입력 지연을 측정했다.
- 스크립트 `experiments/benchmark_conformer_latency.py`, 결과 `reports/conformer-latency-v1/`.
- GPU RTX3090은 워밍업30회 후300회, 서버CPU 2스레드는 워밍업10회 후100회 측정.
  float32·TF32 on, native Fbank부터분류·argmax·CPU라벨반환까지포함. 모델로딩·녹음·통신·잡음제거는제외.
- 중앙값은GPU약15ms·CPU약60~75ms다(두번측정변동). 정확한최종수치/p95/CPU모델은benchmark.json참조.
- 첫1.5초창을모으는대기시간과순수추론지연은별개. 현재는비인과적창전체추론이며스트리밍구현측정이아니다.

## 최신 완료: Conformer 증강만 비교

- 사용자 요청: 보조 특징 없이 기존 증강과 실제 훈련 소음 혼합부터 비교. 중단 후 재개 요청.
- ‘모델 올림’은 새 파일이 보이지 않아 채팅 AI 모델 변경으로 해석한다고 알렸으며, 기존 Conformer를 사용한다.
- 폴더 `experiments/conformer_aug/`, `reports/conformer-aug-v1/`, `runs/conformer-aug-v1/`.
- 무증강4개는 기존 개발CV checkpoint 재사용. legacy(속도+마스킹),noise,combined ×4fold를 새로 학습.
  기존 ASR 원본/같은seed42 head에서 매번 시작,10epoch·LR1e-5/5e-4·WD1e-4·배치32 유지.
- 속도0.9~1.1배는 검수 비패딩 구간 전체 보존, 넘치는 감속 건너뜀, 가속 부족분만0패딩.
  마스킹20~50ms·80mel중3~5bin,훈련에서만native무음값으로 채움. 기존128mel와폭비율을맞춘이식이다.
- 훈련 소음6개: 다른원본의도로·버스·광장·구내식당·믹서·펌프. 기존검증8개원본과분리.
  매입력p.5,SNR U[-5,20]dB,새1.5초구간. silence는600훈련키워드RMS중앙값기준.
- 선택은원본validationF1,이후고정된기존소음검증으로비교. testspk01오디오미사용.
- 새증강검증5개통과. 준비로그download.log/speed_bank.log, 학습로그execution.log.
  실행 `runs/conformer-env/bin/python -m experiments.conformer_aug.run`.
- 각last.pt에서epoch재개. 동결프로토콜/속도캐시생성이후소스수정시새버전을쓸것.
- 12회·120epoch 새 학습 및 기존 무증강4개 재사용 완료. 모든 실행에서 baseline과 초기본체·head해시 동일.
- 개발4fold seed42 macro-F1, 원본/10/0/−5/−10dB:
  - baseline:99.3204/95.4593/72.5709/51.8756/34.5590%.
  - legacy:99.2038/95.6524/73.0936/52.2871/34.3696%.
  - noise:99.2043/96.5309/78.4748/57.6967/38.2219%.
  - combined:98.9765/96.5033/78.0475/57.3530/38.2696%.
- 이번선별은noise가유망:0dB+5.90%p,−10dB+3.66%p,원본−0.12%p. legacy추가효과작고combined추가이득불명확.
- 잡음단독오반응률0/−10dB baseline6.41/11.09%,noise4.22/9.53%.
  원본학습–검증격차baseline0.34%p,noise0.30%p로변화작아‘과적합해결’보다소음강건성개선근거가분명함.
- 기존검증혼합80,512개를SHA256까지일치시켜재사용. 전체예측325,296개,새예측243,972개.
- 독립검사:2,064개조건F1/정확도·120epoch24,360개예측지표재계산,
  새12개최고checkpoint의train/val9,744개·잡음768개재추론일치. 원본/소음/가중치해시·정규화보존.
- 전체테스트48개통과. 73,080개훈련draw를기록했고조건간sampler·speed/noise랜덤스트림짝일치확인.
- 보고서 `reports/conformer-aug-v1/README.md`, `augmentation_comparison.png`, `learning_curves.png`, `VALIDATION.md`.
  재검증 `runs/conformer-env/bin/python -m experiments.check_conformer_aug`.
- 보조음향특징은추가하지않음. 단일seed42의개발4명결과,최고본은원본val로선택. 모든프로세스종료.

## 최신 완료: 강한 실제 환경 소음 평가

- 사용자 요청: 도로·사람 많은 곳·기계 등 다양한 강한 소음, 긴 녹음의 랜덤 구간 활용.
- 작업 브랜치 `pretrain`, `experiments/noise_stress/` 신규 구현. 기존 학습 코드·checkpoint 보존.
- 기존 개발4-fold seed42의최고 모델8개를 고정해평가만 진행. test spk01은 이번에도 추론하지 않는다.
- 프로젝트 전체test미사용이라고 주장하면 안 된다. `jaehun` 기존 선택 모델의 과거test 평가 이력은 있다.
- 소음8종: DEMAND 도로·차량·구내식당·식당·역·세탁기, RPi 공기압축기·환풍기.
  원본 MD5검증 후mono16k, DEMAND는마이크ch01만 사용. 원본 단위로개발평가에예약.
- 1.5초 연속 구간3회, SNR10/0/−5/−10dB. 같은파일·소음·draw는모델/레벨간동일구간.
- full-clip power기준SNR, silence/noise-only는600개훈련키워드RMS중앙값기준. 혼합전체감쇠로클리핑방지.
- 보고서 `reports/noise-stress-v1/README.md`, 다운로드로그 `runs/noise-stress-v1/download.log`.
  평가명령 `runs/conformer-env/bin/python -u -m experiments.noise_stress.run`.
  캐시가 완료한 조건을 건너뛰며 동결 프로토콜 변경을 거부한다. 동결 소스 수정 시 새 버전으로 실행할 것.
- 384개 혼합 조건(모델별768개) 완료. 전체 예측162,648개: 원본1,624·혼합155,904·잡음단독5,120.
- 원본/10/0/−5/−10dB 평균F1: ResNet80.3789/51.4015/32.4382/23.5387/17.8542%,
  Conformer99.3204/95.4593/72.5709/51.8756/34.5590%.
- Conformer−10dB: 차량96.97·세탁기88.67%지만 식당3.31·압축기6.82·역12.96·구내식당13.25%.
  같은SNR이어도 원본의 스펙트럼·시간 특성이 달라 난도가 크게 다름. 군중·압축기에 특히 취약.
- 잡음단독 명령어 오반응률0/−10dB: ResNet18.59/31.72%, Conformer6.41/11.09%.
  실제명령어−10dB Conformer:33.69%정답·46.92%silence누락·19.39%다른명령어.
- 원래 clean 예측1,624개 모두 재현. 독립 검사에서1,032개조건 지표 재계산,
  256개혼합 복원·512개예측 재추론 일치·812개WAV해시·8개checkpoint해시 보존 확인.
  silence기준 훈련600개RMS중앙값도4fold 모두 재계산 일치. 전체 테스트43개 통과.
- `reports/noise-stress-v1/README.md`에 결과·그래프·듣기 예시·출처·한계, `VALIDATION.md`에 검증 기록.
  독립 재검증 `runs/conformer-env/bin/python -m experiments.check_noise_stress`.
- 다음 개선을 한다면 이번8개 원본과 다른 녹음을 훈련 증강용으로 확보한 뒤 같은 개발 스트레스 조건으로 비교.
  이번 결과는 실제 거리·잔향·unknown·최종test 성능을 보장하지 않는다. 모든 평가 프로세스 종료.

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
