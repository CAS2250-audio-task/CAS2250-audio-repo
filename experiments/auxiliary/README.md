# 시간 정렬 음향 특징 실험

기존 `src/`와 ref 설정을 수정하지 않고, `jaehun`에서 1.5초 데이터의 보조 입력 효과를 비교한다.
학습 데이터는 원래 고정 WAV 그대로이며 **잡음/피치 증강은 OFF**다.
모든 체크포인트는 **원본 validation macro F1**으로 선택한다. 소음 평가는 별도 보조 결과다.

## 실험 세트

| 세트 | 스펙트로그램에 추가하는 특징 |
|---|---|
| 0 | 없음: 원래 ResNet18 |
| 1 | log F0, Δlog F0, 유성 여부·확률 |
| 2 | F1, F2, F3 |
| 3 | 2 + B1, B2, B3 |
| 4 | 2 + ΔF1, ΔF2, ΔF3 |
| 5 | flatness, normalized spectral flux, centroid |
| 6 | 프레임별 log RMS |
| 7 | 1~6의 모든 특징, 중복 제거 |

각 특징마다 측정 가능 채널을 붙인다. 이 채널은 **정확도 정답이 아니다**.
수치가 나온 잘못된 포먼트·피치까지 완전히 식별할 수는 없다.

F0는 pYIN(65~600 Hz, 64 ms 창, 10 ms 간격)으로 추정하여 반음으로 표현한다.
유성 확률은 F0 정확도 확률이 아니다. F1~F3/B1~B3는 Praat Burg LPC(5 formants,
공통 ceiling 5500 Hz, 유효 창 25 ms)로 측정하고 **Hz 단위**를 유지한다.
포먼트 ceiling을 화자/정답에 맞춰 사후 조절하지 않는다.
Δ는 5프레임의 국소 기울기이며 중간에 측정 불가 값이 있으면 계산하지 않는다.
Flux는 L1 정규화 magnitude 스펙트럼의 프레임 차이 RMS다. 음량 변화의 영향을 줄이려는
선택이지 소음 불변 특징은 아니다. RMS는 -160 dB 바닥값을 사용하고 무음의 RMS=0을
추출 실패로 취급하지 않는다.

모든 실수 특징의 평균/표준편차는 **train의 관측된 프레임만**으로 적합한다.
확률·불리언은 그대로 사용한다. 변환/표준화 후 결측은 0으로 채우고 가능 표시를 0으로 둔다.
원시 캐시에는 결측을 NaN으로 보존하지만 신경망에 들어가는 배열은 모두 finite여야 한다.
별도 VAD 정답·화자 ID·수동 구간은 특징 추출기 입력으로 사용하지 않는다.

## 모델과 비교 조건

보조 경로: Conv1D 32/64/64, kernel 5, dilation 1/2/4, GroupNorm, GELU, dropout 0.2.
ResNet `layer1` 출력에 같은 시간 중심끼리 결합한다. 기존 stem의 conv7/stride2/pad3와
maxpool3/stride2/pad1 때문에 출력 j의 중심은 입력 4*j다. 배열 길이만 임의로 resize하지 않는다.
보조 채널을 주파수 축에 broadcast하여 붙이고 1x1 projection으로 원래 64채널을 복구한다.
projection은 주 경로 identity/보조 경로 zero로 초기화해 초기 출력이 기준 모델과 같다.
학습 중에는 두 경로 모두 업데이트한다.

뒤쪽 `layer2/3/4 -> avgpool -> 512 -> Linear(11)`은 ref 형태 그대로다.
`forward_features()`는 512차원을 제공해 향후 MLP/멀티태스크 head를 별도 실험할 수 있다.
고정 4구간 풀링은 주 실험에서 사용하지 않는다.

학습률 5e-4, AdamW(weight decay 1e-4), batch 32, 10 epochs, cosine, pretrained ImageNet,
역빈도 WeightedRandomSampler를 유지한다. 같은 seed의 backbone 초기 해시를 비교한다.
0~7 seed42를 먼저 측정하고, 원본 val macro F1 상위 보조 세트 2개 및 기준을 seed43/44로
확인한다. 동률이면 noisy val 평균 F1으로 정한다. 유망한 세트에는 global late-fusion과
추가 파라미터 수를 맞춘 spectral-only residual 대조군도 측정한다.
확인된 세트의 평균 원본 val F1으로 최종 구성을 고르고, seed42를 사전 지정한 test 모델로 쓴다.
기준과 선택 모델의 test 결과를 함께 기록하고 test를 본 뒤 재튜닝하지 않는다.

## 소음 평가

ESC-50의 rain/engine/church_bells/keyboard_typing 4종을 원본 validation의 **복사본**에
20/10/0 dB로 더한다. 학습에는 사용하지 않는다. 매 모델이 동일한 잡음 파일·구간·SNR을 본다.
원래 녹음에도 잡음이 있으므로 SNR은 '기존 full-clip 파워 대 추가 잡음 파워'다.
silence와 noise-only의 레벨은 train keyword full-clip RMS 중앙값을 기준으로 정한 명목 SNR이며,
실제 무음의 음성 대 잡음 SNR이라고 부르지 않는다. peak가 .99를 넘으면 전체 혼합 신호를
같은 비율로 축소해 hard clipping을 피한다.

소음 평가마다 noisy WAV에서 스펙트로그램과 모든 보조 특징을 다시 추출한다.
추가로 category/SNR마다 noise-only 12개를 평가해 단어 오인율을 기록한다.
ESC-50 fold4는 validation, fold5는 final test에 쓰며 원본 source ID를 분리한다.
잡음 manifest에 저장한 고정 commit/URL/Freesound ID/hash로 출처를 재현한다.
데이터셋 라이선스는 CC BY-NC 3.0이며 개인 원본의 출처는 각 Freesound ID를 참고한다.
출처: https://github.com/karolpiczak/ESC-50

## 실행

실제 실행 환경 버전은 study의 environment.json에 기록한다.
기준 requirements의 torch 스택과 `requirements-experiments.txt`가 필요하다.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m experiments.auxiliary.study prepare --workers 4
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m experiments.auxiliary.study screen
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m experiments.auxiliary.study confirm
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m experiments.auxiliary.study finalize --workers 4
```

`runs/auxiliary-v1/`에 캐시·환경·노이즈 출처·가중치·전체 예측을 보관한다.
Git에는 코드와 `reports/auxiliary-v1/`의 요약·곡선·혼동표·오류 목록을 기록한다.
데이터 정답/분할, 원본 및 고정 1.5초 WAV는 변경하지 않는다.
test 특징은 최종 선택을 파일로 고정한 뒤에만 추출한다.
