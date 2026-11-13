# 야간 실험 인계

## 완료 결과

- 한국시간2026-11-13 09:08완료, 시작03:18부터약5시간50분. 신규28fold×90epoch=2,520epoch와최종DFN3비교·독립검사완료. GPU유휴. 실패표식없음.
- 선택: WD1e-4·Dropout0.2·기존소음+포먼트B·보조없음·무필터. 0dB F1 91.2974%, 정확도91.8719%. 이전89.7492/90.1888% 대비+1.5482/+1.6831%p.
- WD1e-3은89.6436%,Dropout0.2는90.6651%. 선택Dropout에서A90.6826%,B91.2974%,A+B90.8539%. 둘다통과한정규화후보가아니므로WD+Dropout결합은규칙대로생략했다.
- 보조3은91.3015%로B대비+0.00408%p에그쳐채택기준+0.5%p를못넘었다. 보조7은90.9084%. DFN3는90.4063%로하락했다.
- 02의F1 76.6339→79.3503→81.9513%. 추가draw1~2의B F1도89.7523→91.2724%. 피치단독효과나성별인과효과를분리하는실험은아니다.
- 검증: epoch예측4,604,040개·훈련draw1,534,680개·최종146,580개재계산, 최고본51,156개재추론일치. source해시보존과전체마지막epoch100확인. test01미사용.
- 진입점 `reports/overnight-v1/RESULTS.md`. 독립검사 `independent_audit.json`,선택재집계 `review_validation.json`,화자별 `fold_results.csv`,추가구간 `draw_split_summary.csv`. 완료후재실행하지않는다.

브랜치 `overnight-regularization-formant-aux`, worktree `/root/audio/.worktrees/overnight`.
기준 commit은 pretrain의 `9fbdad4`이며 기존 실험·체크포인트는 읽기만 한다.

## 확정한 흐름

1. 기존 소음 혼합·batch16 기준을 재사용한다. WD1e-4→1e-3, 인코더 Dropout0.1→0.2를 각각 비교하고 둘 다 선택 기준을 통과할 때 결합한다.
2. 선택한 정규화에서 A=SpecAugment+속도+피치, B=포먼트 한 설정, A+B를 비교한다. 기존 소음p0.5·SNR[-5,20]은 모든 조건에 공통이다. 각 증강의 적용 확률은0.5로 독립이다.
3. 선택한 증강에서 보조 세트0·3·7을 비교한다. 세트3은 포먼트+대역폭,7은17개 전체 특징이다. 입력은 실제 증강·혼합 파형에서 추출하고 scaler는 각fold 원본train에만 적합한다.
4. 선택한 동일 모델에 DFN3 off/on을 비교한다. 필터 뒤의 파형에서 보조 특징도 재추출한다.

4-fold(spk02~05),seed42,batch16,609draw/epoch,전체100epoch,조기종료없음.
기존noise epoch10 LAST에서 모델·AdamW·Torch/CUDA/sampler RNG를 복원하며 후보 변화는epoch11부터 적용한다.
인코더 전체 학습,LR1e-5/head5e-4 유지. 앞단 고정·동결·추가 LR 탐색은 제외한다.
최고epoch는 무필터 고정0dB draw0 F1, 최종 보고는 전체3draw이다.
조건 선택은4fold평균F1 +0.5%p 이상·최저화자하락1%p 이내의 이전 기준을 쓴다.
test01은 사용하지 않는다. 결과는 개발 평가다.

## 증강 세부

- 피치: 파일내0/±1/±1/0반음anchor 변화, 내부anchor차이≥0.35. 기존RubberBand2 R2·포먼트보존·startup지연보정 정책.
- 속도: 피치유지0.9~1.1배. 기존검수비패딩구간을보호하며넘치는감속은건너뛰고가속부족분은좌우0패딩.
- SpecAugment: 시간20~50ms·주파수3~5/80mel bin. 시간마스크는 보조값·availability에도 적용한다.
- 포먼트: 비율1/1.1~1.1의로그균등분포. 피치중앙값·범위·길이를유지하고 RMS정렬·peak0.99제한.
- 순서: 피치→속도→포먼트→실제소음→스펙트로그램마스킹. 파형활성조합7종마다원본별2개고정변형을만든다.

## 진행 확인과 재개

- 정리: `reports/overnight-v1/README.md`
- 상태: `runs/overnight-v1/progress.json`
- 로그: `runs/overnight-v1/execution.log`
- 프로세스: `runs/overnight-v1/launch.json`, `process.json`
- 완료: `runs/overnight-v1/complete.json`, `reports/overnight-v1/independent_audit.json`
- 실패: `runs/overnight-v1/failure.json`

실행 Python은 `/root/audio/.worktrees/pretrain/runs/denoise-env/bin/python`을 공유한다.
재개 명령은 해당 Python으로 `-u -m experiments.overnight.run`이며 먼저process/lock/log를확인한다.
백그라운드에서stdin을닫고새session으로실행한다. VSCode/SSH연결을닫아도서버와GPU할당이살아있으면계속돈다.
각epoch LAST에재개상태를저장한다. 완료fold는예측재계산·최고본재추론검사후새로만든큰optimizer LAST만삭제하고best와모든epoch기록을보존한다.
protocol생성이후`experiments/overnight`소스를수정하지않는다.

## 시작 전 확인

4개핵심검사통과. native logits 보존·보조역전파·7종파형조합유한성·1.5초보존·합성음포먼트변화/피치보존·시간마스크정렬을GPU에서확인했다.
합성음테스트의확률마스킹배치를2→16으로늘려실제마스크를확인했다. 학습정책은변경하지않았다.
이전초안의준비파일은`reports/overnight-v1/preparation_v0/`에보관하며그초안으로학습한실행은없다.
