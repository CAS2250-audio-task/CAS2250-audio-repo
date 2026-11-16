# 전 대역 유지·저주파 해상도 배분 비교

전 대역20~7800Hz를 유지하면서64mel 중 저주파에 배정하는 비율만 조금 늘린다.
고주파를 자르거나 특정 대역의 attention 가중치를 수동으로 강제하지 않는다.
standard는 기존torchaudio frontend와 bitwise 동일하다.
lowfocus는 정규화된mel좌표u를0.8u+0.2u²로 재배치한다.
1kHz 미만 중심필터는22→25개,4kHz 초과는15→13개이며 빈 필터는 없다.
각 조건/fold의 원본 학습 데이터로만 평균·표준편차를 새로 적합한다.

## 학습 범위

Pooled로 고정한다. 집계 종류 자체를 비교하는 실험은 아니다.

| 구조 | standard | lowfocus | 검증 화자 |
|---|---|---|---|
| 기존 local CNN | 실행 | 실행 |03 |
| 축 분리형 소형2D + local | 실행 | 실행 |03 |
| 세로CNN, 중간/최종 주파수2구역 보존 | 실행 | 실행 |03 |
| 위 세로CNN + 세로 self-attention | 실행 | 실행 |03 |
| 축 분리형 + 세로 self-attention | 실행 | 실행 |03 |
| 기존 전체0dB 최고 local Pooled | 실행 | 미실행 |02 |

총11조건. 여성02는 기존 최고 구조의 표준 입력 기준을 확인하는 별도 실험이며,
여성에서 저주파 재배치가 개선됐다고 판단할 수 있는 비교는 이번 범위에 포함하지 않는다.
화자03 실험은02/04/05로 학습, 화자02 실험은03/04/05로 학습한다. test01은 읽지 않는다.

사용자 선택은500epoch. 계획1000epoch cosine의 첫500을 학습하고 최종3draw 평가를 수행한다.
optimizer/projector/정규화/선택best를 보존하여 동일 프로토콜로1000까지 연장할 수 있다.
기존 완료 실험/quick-frequency40 trainer는 수정하지 않았고 별도 trainer를 사용한다.

```bash
python -m experiments.low_frequency_grid.launch --stop-after 500
python -m experiments.low_frequency_grid.monitor --watch
# 감독 프로세스 종료 후, 전체11조건을1000까지 연장할 때:
python -m experiments.low_frequency_grid.launch --resume --stop-after 1000 --limit-minutes 300
```

500조건이 완료됐어도1000연장 시 checkpointepoch를 검사해 이어서 학습한다.
런타임 stop budget은 고정프로토콜에 넣지 않고 progress/summary에 기록한다.
중단 후 재개도 --resume으로 가능하다. 실행 중 중복감독 프로세스를 시작하지 않는다.
소스를 수정하면 재개가 거부될 수 있다.

이것은 검출기 전 고정1.5초 입력의 frontend 비교다. 검출 후 가변길이 성능으로 일반화하지 않는다.
낮은대역 확대는 새 정보를 생성하지 않고 주파수 해상도 배분을 바꾸는 가설이다.
