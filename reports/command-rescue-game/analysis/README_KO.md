# 보고서용 게임 반복 실험

프로젝트 루트에서 실행한다. Node와 Python matplotlib/numpy가 필요하다.

```sh
node analysis/run-stats.mjs --games 10000 --seed 42 --out reports/game-statistics
```

이 환경에서는 `node` 대신 `/root/CAS2250-artifacts/tools/node-v22.16.0-linux-x64/bin/node`를 사용한다.
`--help`로 전체 옵션 확인. 통계만 생성하려면 `--no-plots`, 기본 시나리오의 라운드별 원시 CSV도 원하면 `--round-logs`.
기존 결과 폴더는 덮어쓰지 않으므로 다시 실행할 때 다른 `--out`을 지정한다.

기본 실험은 10음향조건 × 6설정(기본·제한시간 4추가·지연배율 ×1) × 조건당 1만 게임 = 60만 게임이다.
기본 설정의 출격은 10만 게임이며 민감도 분석과 혼합해 하나의 생존율로 집계하지 않는다.
최대 20라운드/실패3회, 제한 3200ms, 처리 지연 ×100, 자동 반응 중앙값450ms/σ0.35를 사용한다.
현재 사이트의 개인 설정이나 친구의 플레이 기록은 수집하지 않는다.

- `REPORT_KO.md`: 방법·기본 결과표·그림 설명·해석 범위와 보고서 문구.
- `summary.csv/json`: 조건별 통계, 95% 조건부 Monte Carlo 구간과 수식 검산값.
- `games.csv`: 모든 시나리오의 게임별 생존·실패·진행 라운드.
- `bank-conditions.csv`, `bank-commands.csv`: 고정 bank 자체 인식 집계(조건당30개/명령당3개).
- PNG/PDF 4종: 생존율, 제한시간 민감도, 실패 플래그, 처리 지연 배율 비교.

시뮬레이션은 사이트의 실제 `dist/core.mjs`를 그대로 import한다. 음성 bank의 실제 사전 추론 결과를 재표집하며 정확도를 난수로 조작하지 않는다.
이 결과는 spk03 개발검증 음성에 조건부인 게임 실험이다. 검출기는 미연결이고 자동 반응시간은 가정값이다.
95% 구간은 게임 반복의 Monte Carlo 변동만 표현한다. 독립 test 성능·사람 실측·모델 일반화 신뢰구간으로 쓰지 않는다.
동일 seed를 조건 간 재사용하므로 조건 간 결과는 짝지어진 비교다. 인식/시간 오류 플래그는 중복될 수 있다.

검증: `node --test checks/statistics.test.mjs`.
결과를 다시 그릴 때: `python analysis/plot_stats.py reports/game-statistics`.
