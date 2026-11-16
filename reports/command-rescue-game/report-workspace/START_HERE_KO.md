# 보고서 담당자 작업 공간

게임 통계 실험을 보고서에 옮기기 위한 자료와 초안이다. 전체 검출기/분류기 보고서의 본 실험과 구분해, 시연 또는 응용 사례 절에 넣는다.

1. `NEW_CHAT_PROMPT_KO.md`를 새 채팅의 첫 메시지로 붙여 넣는다.
2. `DRAFT_KO.md`에서 미리 작성한 방법·결과·한계·그림 설명을 다듬는다.
3. `WORK_NOTES_KO.md`에 연결할 절 번호, 요청 사항, 확인한 수치, 남은 결정을 적는다.
4. 수치의 기준은 `../reports/game-statistics-20261116/summary.json`과 CSV다. 초안 문구와 충돌하면 수치를 확인하고 초안을 고친다.

이 공간은 편집 가능한 보고서 작업 공간이다. 자동 생성 결과 파일을 문장 편집용으로 사용하지 않는다.

## 근거 자료

| 목적 | 경로 |
|---|---|
| 자동 생성 설명과 기본 결과표 | `../reports/game-statistics-20261116/REPORT_KO.md` |
| 조건별 수치·설정·모델/소음 출처·코드 해시 | `../reports/game-statistics-20261116/summary.json` |
| 표 편집 | `../reports/game-statistics-20261116/summary.csv` |
| 고정 bank 인식률 (조건당30개) | `../reports/game-statistics-20261116/bank-conditions.csv` |
| 명령별 결과 (조건·명령당3개) | `../reports/game-statistics-20261116/bank-commands.csv` |
| 60만 게임 원시 집계 | `../reports/game-statistics-20261116/games.csv.gz` |
| 그림 4종 (PNG/PDF) | `../reports/game-statistics-20261116/` |
| 통계 재현 방법 | `../analysis/README_KO.md` |
| 검증 기록 | `../reports/game-statistics-20261116/VERIFICATION.json` |
| 결과 파일 무결성 | `../reports/game-statistics-20261116/ARTIFACTS.json` |

게임 링크: https://command-rescue-cas2250.yellowdove22.chatgpt.site (링크 공개).

## 담당 범위

현재 모델은 spk03 개발 검증 기반의 기존 분류기, 검출기 미연결이다. 시뮬레이션은 고정 bank를 재표집하고 가정한 로그정규 반응시간을 사용한다. 사이트 친구들의 플레이 기록을 수집한 결과가 아니다. ‘60만 개 새 음성 평가’, ‘실제 사람 실험’, ‘독립 test’, ‘전체 검출기 포함 성능’으로 쓰지 않는다.

기본 결과는 10조건×1만 게임=10만 게임이다. 나머지 50만 게임은 제한시간 민감도 및 처리 지연 ×1 비교다. 서로 다른 설정의 결과를 하나의 평균 생존율로 섞지 않는다.

팀 GitHub의 `docs/command-rescue-report-20261116` 브랜치, `reports/command-rescue-game/`에 이 작업 공간과 통계 코드·자료를 함께 저장했다. 이 브랜치는 팀 main의 aaa535dad6b35ce3de85ebdb621ae765ebcc74a4를 기준으로 한다. 게임 사이트용 저장소의 fbaa90a와는 별개의 GitHub 커밋이다. 웹페이지 변경 작업은 포함하지 않는다.
