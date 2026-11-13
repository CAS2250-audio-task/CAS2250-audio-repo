# Attention 마스킹·발화 구간 피치·경량화 실험

브랜치 `attention-pitch-v1`, worktree `/root/audio/.worktrees/attention-pitch`.
이전 완료 실험은 `/root/audio/.worktrees/overnight`에 그대로 보존한다.

## 진행과 결과

- 진행: `runs/attention-pitch-v1/progress.json`
- 실행 로그: `runs/attention-pitch-v1/execution.log`
- 보고서: `reports/attention-pitch-v1/README.md`
- 화자별 결과: `reports/attention-pitch-v1/fold_results.csv`
- 학습곡선: `reports/attention-pitch-v1/speaker_curves.png`
- 오류 종료: `runs/attention-pitch-v1/failure.json`
- 전체 완료: `runs/attention-pitch-v1/complete.json`
- 새 피치 청취: `reports/attention-pitch-v1/listening/README.md`

```bash
tail -f /root/audio/.worktrees/attention-pitch/runs/attention-pitch-v1/execution.log
```

## 고정 범위

기준은 이전 선택본의 noise + weak formant + dropout0.2, WD1e-4, 보조0, 필터없음이다.
원본50%·소음50%, 소음SNR[-5,20], 기존6훈련원본을 유지한다.

같은 noise epoch10 LAST에서 모델·AdamW·Torch/CUDA/샘플러 RNG를 복원하고11~100epoch를 학습한다.
batch16,609draw/epoch,encoder1e-5/head5e-4, 전체 인코더 학습, 조기 종료 없음.
기준은 기존 결과를 재사용하되 원래 예측과 학습draw의 정확한 일치를 사전 검사한다.

1. 기준: 마스킹X·새피치X.
2. attention: 실제 입력에서 추정한 VAD 확률로 모든 블록의 key에 log(g) 편향, 최종시간축도 g가중평균.
3. pitch: 발화 내부의 부드러운 다중 변곡 피치.
4. attention_pitch: 두 변경 결합.
5. 채택된12블록 조건을8블록·6블록으로 축소. 시간 여유가 있을 때만 실행.

마스킹: 고정된 공식 Silero VAD 가중치, 입력1.5초마다 상태 초기화,32ms창 확률을40ms CNN 위치에 정렬.
확률 평활화와 앞뒤80ms 확장 후g=.05+.95p, 최대확률0.3미만이면 전체 유지한다.
원본 경계/정답/화자 ID는 실제 모델의 VAD 입력으로 주지 않는다. VAD는 분류기 파라미터나 optimizer에 들어가지 않는다.

피치: 기존 원본 자동VAD 경계를1.5초 좌표에 옮겨 발화 내부에3~4개의 무작위 내부 앵커를 둔다.
PCHIP 보간·변화속도32반음/초 제한,8변형/파일, 선택확률50%다.
Praat의 원본 pitch tier에 시간별 배율을 적용하고 overlap-add로 재합성한다.
합성 모음에서 실제 F0와 요구한 곡선의 오차·상관을 검사한다.
새 피치 단계에서 발화 바깥은 원본과 비트 단위로 동일하며 경계20ms는 안쪽에서 크로스페이드한다.
이후 기존 공통 포먼트 변환은 이전과 같은 전체파형 방식으로 적용한다.
추가 소음은 두 변형 뒤에 섞어 피치를 바꾸지 않는다. 발화와 겹친 기존 녹음 배경은 독립 분리하지 않는다.
speed·SpecAugment·보조특징·잡음제거·새WD/LR탐색은 이번 범위에 없다.

## 평가와 해석

개발02~05의4fold이며 각각 나머지3명609개 학습·203개 검증이다. test01 음성은 읽거나 평가하지 않는다.
epoch 선택은 기존 고정0dB draw0, 최종은8소음×3draw 및원본·잡음단독으로 동일하다.
평균0dB F1 +0.5%p 이상·개별화자 하락1%p 이내일 때 변경을 채택하고 raw 최고점도 따로 표시한다.
경량화는 평균 F1 하락0.5%p 이내·개별화자 하락1%p 이내·원본정확도 평균 하락0.5%p 이내 모델 중 가장 작은 것을 표시한다.

원본 자동VAD 경계를 평가에 주는 비교는 진단 전용이다. 수동으로 확정한 정답 경계나 실사용 성능이라고 주장하지 않는다.
진단은 원본경계/자동검출 각각 pool-only와attention+pool을 비교하고 조건 선택에는 쓰지 않는다.
경량화는 원본0~11에서 균등한 깊이의8/6블록을 선택해 파라미터와 해당 AdamW 상태를 보존한다.
소형 모델도 같은epoch10출발·100epoch예산이다. 교사 증류나 추가학습의 효과와 섞지 않는다.

## 재개와 보존

인터럽트 시 각fold의 `last.pt`에서 다음epoch부터 재개한다. 끝난fold는 다시 학습하지 않는다.
단독실행 확인은 `driver.lock`, 프로세스 정보는 `process.json`/`launch.json`에 남긴다.

```bash
cd /root/audio/.worktrees/attention-pitch
/root/audio/.worktrees/pretrain/runs/denoise-env/bin/python -m experiments.launch_attention_pitch
```

실행 중 `experiments/attention_pitch/`와 pin된 과거 소스를 수정하면 안 된다.
모든 best체크포인트·학습곡선·epoch예측·학습draw를 보존한다.
공간 관리를 위해 새 실험에서 완전히 끝나 검증을 통과한fold의 LAST optimizer만 지운다.
7시간은 선택 경량화의 시간 제한이며 필수 core나 진행 중fold를 중간에 자르지 않는다.

SSH/VSCode 연결은 종료해도 되지만 서버·컨테이너·GPU 할당은 유지해야 한다.

## 이번 실행 확인

2026-11-13 14:51 KST 시작. PID/SID `259650`, 부모PID1, SIGHUP무시, 표준입력DEVNULL.
19epoch LAST와 optimizer 551step 및Torch/CUDA/샘플러 RNG 저장을 확인했다. 소스 커밋 `2faebd6`.
기준·마스킹·새피치·결합 비교 후 선택 조건의8/6블록을 시간 여유에 따라 실행한다.
