# 다음 채팅 인계: 두 서버의 검출기 개선

사용자 목표: 본 서버와 Jupyter 서버에서 검출기를 개선한다. 다음 후보는 화자 증강과 추가 증강(마스킹 등), 보조 입력, 모델 구조 개선이다. 우선 현재 실험 결과를 확인한 뒤 다음 실험을 정한다. 새 학습을 중복 실행하지 않는다.

## 현재 상태

- 브랜치: experiment/detector-diversity-v2. 로컬 경로 /root/CAS2250-detector-diversity-v2.
- Python: /root/CAS2250-resnet18-pretrain-100/.venv/bin/python.
- 로컬 baseline/lookahead 각각300epoch 완료. best163/152. 0dB 발화 내부 손실21.88%/13.23%, eventF1 75.59%/69.62%, noise-only 오반응26/80 vs62/80. 원본eventF1 96.60%/75.52%. 단순히 미래문맥을 채택하기에는 추가오검출 문제가 남는다.
- Jupyter boundary/spectral의300epoch 결과 수신 완료. best210/183. 0dB 발화내부손실20.00%/20.03%, 검출F1 76.34%/73.12%, noise-only오반응29/80으로동일. 각모델checkpoint와결과가로컬에수집됐다.
- 공개 릴리스 detector-diversity-v2-distributed-assets-v1에 jupyter-boundary-spectral-results.tar.gz(.sha256)가업로드됐다. 미래문맥+경계헤드조합300epoch 재검증이자동진행중이다. 현재진행은 distributed.json 및 combined/progress.json을확인한다. 4종비교와새환경audit는 RESULTS.md 참조.
- scripts/detector_diversity_collect.py가 runs/detector-diversity-v2/distributed.json을 갱신하며 결과를 기다린다. 수신 후 전체비교→유용한변경 조합/재검증→보고서업로드가 자동설정돼 있다. 유용함은 baseline보다 기존검증 비대칭시간비용1% 이상 개선. 새환경audit를 선택에 쓰지 않는다.
- 새 학습 전 distributed.json, collector.log, 실제 프로세스를 확인하고 수집기와 중복 실행하지 않는다.

## 실험 조건

새 훈련소음88원본(ESC-50의22환경유형,4원본씩)+기존6=94. 별도22원본은새환경audit. src_file 원본분리. 기존검증8개는학습금지. ESC-50 라이선스/출처는 noise_provenance.json과assets내LICENSE.

훈련: S03 곽민채/S04 최환준/S05 이서준600개. 검증S02 노정민200개(과거spk03와같은화자). S01 강재훈은평가하지않았다. 정답은 speech_start/end_in_trimmed_s로원음좌표를사용한다.

증강: 발화부기준SNR, 원본20%/5~20dB40%/0~5dB30%/-5~0dB10%(양성window기준), source-family샘플링, 일부2소음혼합/느린소음레벨변화/약한EQ·에너지곡선/전체gain. 피치·포먼트·속도·RIR·마스킹은이번v2에없다. 순간소음의발화부가완전무음이면SNR계산을위해재추출한다.

공통AdamW1e-3 cosine1e-5,batch32,seed42,300epoch,5ms hop. 프레임BCE 앞50ms 양성가중3/뒤50ms5. 모델선택비용은 앞누락3/뒤·내부·완전누락5/외부포함1. boundary head는훈련보조이며decoder는공통. lookahead는80ms,frontend lookahead약12.5ms별도. 모든기준선/변형은동일증강정책이다.

## 출력과 평가 유의점

모델은5ms별speech logit[B,T],sigmoid후공통decode(.6/.4,20/50ms확인후backdate)로0개이상의원음초단위구간을출력한다. 여유분0. 여러구간자동한개선택/고정1.5초변환은아직없다. 시작끝head는기본출력에사용하지않는다.

v1 predict.py는v2체크포인트와호환되지않으며margin옵션을줘도JSON에는raw경계만기록한다. 후단재학습용v2exporter는raw/crop경계,앞뒤margin,원본ID,미검출/복수검출정책을명시해야한다. 원음주변소리를추가하는margin과zero-padding을구분한다. margin은검증에서앞30/50/80ms,뒤20/30/50ms등을비교할초안이고미적용이다.

평가는발화내누락비율과외부포함시간/noise-only오반응을함께본다. loss만최소화하면전체파일출력도유리하다. 시간이잘린비율과음소정보손실은같지않다. 파찰음/마찰음손실은현재가설이며음소정렬·청취로확정하지않았다.

## 다음 실험 후보와 읽을 자료

1. 기존 화자증강 설계(파일별 큰기본값+파일내 작은완만한곡선,시간맵과라벨동기화): https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/experiment/compact-local-acoustic-aux/docs/CONTINUOUS_SPEAKER_AUGMENTATION_PLAN.md
2. 과거 실제적용 증강과 취소된계획 구분: https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/experiment/compact-local-acoustic-aux/docs/AUGMENTATION_HISTORY.md
3. 검출기개선 설계(미래문맥,2DCNN,경계head,배경통계·자음특징 보조입력,원본/noisy일관성): https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/experiment/speech-boundary-detector/reports/boundary-detector-v1/DETECTOR_DIVERSITY_AND_DESIGN_PLAN.md
4. v1 오답/같은입력의분류기 비교: https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/experiment/speech-boundary-detector/reports/boundary-detector-v1/error-types/README.md
5. 발화내시간손실의정의/통계: https://github.com/CAS2250-audio-task/CAS2250-audio-repo/blob/experiment/speech-boundary-detector/reports/boundary-detector-v1/error-types/SPEECH_REGION_LOSS.md

보조입력은기존F1/F2/F3/BW를무조건추가하지않는다. 배경대비대역에너지,시간差分/flux/고주파비율을분리대조한다. 정답경계로배경만골라주는oracle입력은배포에서재현되지않는다. 마스킹은약한감쇠/완전가림을분리하고경계핵심단서를통째로지우지않도록검토한다. 시간변형시주석과보조availability까지정렬한다. 기존녹음은추가소음전일뿐무잡음이보장되지않는다.
