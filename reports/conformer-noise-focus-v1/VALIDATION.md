# 검증 및 재개 안내

학습은 `pretrain` 별도 작업 폴더 `/root/audio/.worktrees/pretrain`에서 진행했다. 모델/데이터/이전 동결 실험 코드를 변경하지 않고 새 폴더를 사용했다.

## 프로토콜과 입력

- 원본 ref mono16kHz·1.5초, 개발 화자02~05의4fold. fold당훈련609개/검증203개. test01오디오는읽거나추론하지않는다.
- seed42·10epoch·609draw/epoch와동일초기ASR/분류층·학습률·정규화·clean-best checkpoint 선택을유지한다.
- 기준은 기존 소음 혼합 모델4개 재사용. 추가3조건×4fold=12회/120epoch. 새로운train draw73,080개,기준포함97,440개.
- 모든조건에서원본50%를유지한다. focus는혼합50%중절반U[-5,20],절반U[-3,3]. 총학습량고정으로추가증강이일부기존소음draw를대체한다.
- speech는혼합draw의절반을기존6소음,절반을추가37파일에서뽑는다. 웅성거림/옆사람말소리/방송 유형균등선택후파일균등선택.
- 추가파일은공식MS-SNSD train의Git revision/blobSHA와원본/준비/PCM해시로고정했다. MS-SNSD의DEMAND이름변경파일은사용하지않았다. 원본녹음세션의독립성과배경발화전사는완전히검수하지않았다.
- 평가8원본과파일명/원본/준비해시중복없음. 처음/중간/끝8초를200Hz상관도로검색해뚜렷한복제없음. 압축기와의저주파상관5사례는300~5000Hz에서0.006~0.029로확인했다. 모든부분중복부재를증명하는검사는아니다.

## 정식 평가

- 학습4조건×DFN3 off/on×기존원본/8소음/4SNR/3draw/잡음단독. 총650,592예측. 실제입력81,324개.
- 기존혼합80,512개를SHA256까지일치시킨다. DFN3는혼합파형만받고,원본정답이나분리된잡음을보지않는다.
- 이전DFN3의원본/0dB출력20,940개를재사용한다. 나머지60,384개는같은필터로생성했다. DFN3는훈련하지않고추론때만적용했다.
- 정식분류batch32,DFN3batch16. 큰batch203속도검사중spk03/cafeteria/−5dB에서기존예측차이가발견돼그결과는ignored `batch203_engineering_probe/`에분리했다. 정식결과에는포함하지않는다.
- 최초평가의누락된Path import를수정한뒤재개했다. 학습checkpoint를재생성하지않았다. 이후0dB전체4fold를우선처리한뒤나머지SNR을처리하도록실행순서만바꿨다. 완료단위/DFN캐시는재사용한다.
- 같은학습조건의DFN3 off/on에는동일checkpoint를사용한다. 소음/DFN3결과로epoch나임계값/원본혼합비율을다시고르지않는다.

## 독립 검사

실제완료수치는 [independent_audit.json](independent_audit.json), 실행·산출물해시는 [execution_audit.json](execution_audit.json)에기록한다.

- 4,128개조건의정확도/macro-F1/잡음오반응률을원시예측에서재계산한다.
- 120epoch의검증24,360개예측지표와최초최고epoch선택을확인한다.
- 새12모델의원본train/val9,744개를재추론하고,초기값/정규화/최고checkpoint해시를확인한다.
- 조건간97,440개훈련draw의sampler경로와기본noise선택을짝검사한다. speech/focus스트림의독립성및원본/미변경입력의기존baseline일치를확인한다.
- 훈련입력288개를재구성해SHA256을대조한다.
- 원본/0dB식당/−10dB압축기/잡음단독의DFN파형2,516개와분류20,128개를재검사한다. 재추론도정식평가와같은전체unit및batch32경계를유지한다. 처음16개만별도배치로추론하면경계샘플의예측차이가날수있어전체unit으로맞췄다. 이전DFN캐시재생성도원래전체입력배열의batch16경계에맞춰파형을정확히대조한다.
- 프로젝트테스트54개통과(기존51+새증강3). 새검사는flags-off의기존입력정확재현,정책독립성,원본/넓은SNR유지·1.5초·재현성·clipping방지를확인한다.

## 현재 작업 폴더에서 재개

중복실행을피하려면먼저 `runs/conformer-noise-focus-v1/evaluation_process.json`의PID/명령과 `execution.log`를확인한다. 최초driver의 `process.json`은이미종료된프로세스다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 runs/denoise-env/bin/python -u -m experiments.conformer_noise_focus.run
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 runs/denoise-env/bin/python -u -m experiments.evaluate_conformer_noise_focus
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 runs/denoise-env/bin/python -m experiments.report_conformer_noise_focus
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 runs/denoise-env/bin/python -m experiments.check_conformer_noise_focus
```

완료학습/평가단위를재사용한다. 동결training소스/소음정책/조건을바꾸면새study로진행한다. source준비/중복점검을재실행해이미고정한manifest를덮어쓰지않는다.

대형가중치/파형/DFN캐시/듣기WAV는ignored `runs/`에있으며코드·보고서·예측만Git에보존한다. 듣기샘플은 `experiments.listen_conformer_noise_focus`로내보낸다.
