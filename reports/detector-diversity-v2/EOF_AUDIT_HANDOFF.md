# 우선 작업: EOF 평가 오류 재분석 (미완료)

2026-11-16. 기존 RESULTS.md/NEXT_CHAT_HANDOFF.md의 미래문맥 모델 순위 결론은 아래 평가 오류를 해결한 뒤 재검토해야 한다.

- 모든300epoch 학습/기존평가/업로드는완료. 새학습하지말고고정체크포인트로재평가부터한다.
- 학습augment.py는1.5초target+실제미래80ms를모델에주고처음301frame만loss계산한다.
- models.py는lookahead에서입력끝을F.pad(...,0)로채운다. 입력은표준화된logmel이므로0은무음이아니라평균특징이다. 평가의가변길이batch도짧은샘플뒤를같은0으로채운다. 따라서EOF처리를샘플별길이에맞게고쳐야한다.
- 저장예측의noise-only80개: baseline26개오반응. lookahead62개/70events 중41events가끝80ms에시작,46개가EOF에닿음. combined75개/99events 중69개가끝80ms시작,72개가EOF에닿음.
- 재학습없이마지막mel프레임을80ms유지한빠른noise-only진단: lookahead62→28/80, combined75→30/80. 정식전체재평가가아니므로새성능확정값으로쓰지말것. 기존77.5%/93.8%만으로학습실패·조합실패라고단정한결론은보류한다.

## 파일과 현재 중단 지점

작업경로 /root/CAS2250-detector-diversity-v2, 브랜치 experiment/detector-diversity-v2.
Python /root/CAS2250-resnet18-pretrain-100/.venv/bin/python.

- experiments/detector_diversity/inference.py: 샘플별length를받는EOF hold/silence adapter, 원래scoring프레임삭제없음.
- experiments/check_detector_eof.py: causal출력불변/길이혼합batch와single일치/prefix불변/shape검증3개통과.
- experiments/audit_detector_eof.py: 재평가초안. 아직정상완료하지않았음!
- runs/detector-diversity-v2/eof_audit.log: baseline hold 평가후 lookahead에서prefix변화<1e-4 assertion 실패. 원인을확인해야하며assert만제거하고통과로보고하지말것.
- 추가로실제4원본의첫32case batch에서raw/holdprefix비교는최대차이0이었다. 어떤길이/batch에서첫assert가실패했는지 case별max와위치를출력해확인할것. CUDA sequence길이변경수치차이인지실제context처리문제인지미확정이다.
- reports/detector-diversity-v2/eof-audit/metrics.json은baseline1행만있는불완전중간파일일수있다. 정상완료audit.json이없으면완료로취급하지말것.

## 다음 확인 순서

1. 관련프로세스와로그확인. 새학습불필요. best.pt/last.pt는유지.
2. 같은가중치/같은mixture/동일threshold .6/.4,confirmation20/50ms,margin0 고정.
3. raw정규화0패딩, 마지막특징유지, 실제waveform0패딩후frontend의차이를비교. normalized-logmel의0과물리적무음을구분. 파일별실제길이뒤에context를붙여batchpadding문제도처리.
4. noise-only는긴원본에서동일한앞1.5초+실제다음80~100ms를제공하는대조를실행. audit초안 real_future 함수가앞1.5초파형동일성을assert한다. 이것은아직실행완료되지않았다.
5. 기존8소음0/5/10dB,새22소음0dB,원본200개에서전체재평가. EOF점수프레임을버려수치를좋게만들지말것. threshold/margin/최고epoch를동시에바꾸지말것.
6. 원본결과보존,보정결과별도기록하고모델추천/인계문갱신. 고정best는잘못된이전평가기준으로선택됐다는한계도기록.

소음증강실제설정: 4windows/original중1개(25%)는noise-only,라벨전부0. 나머지3개는80%소음혼합. 양성window기준원본20%,5~20dB40%,0~5dB30%,-5~0dB10%. 94소음원본/family샘플링. noise에p.3로3제어점±3dB느린레벨곡선, p.2로2소음중첩, 혼합전체p.5약한EQ±3dB/p.3느린gain±2dB,마지막전체gain-6~+3dB. 프레임마다소음/SNR을다시뽑는것은아니다. sourcecrop/epoch마다변한다. 낮은에너지발화경계의라벨정확성과변형과다여부는평가오류와분리해검토할것.
