# 주파수 구조18조건 통합 결과

기존 local3개 + 이 서버13개 + 다른 서버Self2개를 통합했다. 모든 조건의 선택 checkpoint 예측에서 원본/0dB F1을 재계산했다.

| CNN | 세로 어텐션 | Mean 0dB F1 | Pooled 0dB F1 | Self 0dB F1 |
|---|---|---:|---:|---:|
| basic | 없음 | 85.44% | 86.77% | 85.55% |
| basic | 있음 | 79.83% | 85.54% | 86.13% |
| lowcost | 없음 | 82.88% | 86.66% | 86.02% |
| lowcost | 있음 | 80.99% | 84.76% | 84.99% |
| frequency | 없음 | 77.03% | 81.38% | 79.92% |
| frequency | 있음 | 81.64% | 85.02% | 82.01% |

전체11종 macro F1과 정답률·명령어10종F1·silenceF1의 순위는 다를 수 있다. 전체F1만으로 명령어 구별의 개선을 단정하지 않는다. 상세값은 all18_comparison.csv에 함께 기록했다.

화자03/seed42 개발검증이며 독립test01 결과가 아니다. 모델별 CNN은 독립 학습했고 선택 epoch도 다르다. CNN구조 변경에는 용량·정규화·문맥 차이도 포함된다.

## 병합·복원 확인

- 원격 balanced-self2 결과와 acoustic-aux 브랜치를 compact-pooling-v1에 병합했다.
- 팀원 릴리스1353개 파일 SHA256, Self2개 각1000epoch, strict checkpoint load,1330개 공통 증강trace hash를 확인했다.
- lowcost Self의 UTF-8 출력 오류는 학습 후README 쓰기 단계였고, 복구 전후 가중치·이력·예측 파일 hash가 같음을 확인했다.
- 다른 서버의 BFDE 소스 표기는 설계문서 커밋을 포함한 checkout 기록이며 학습 소스파일 hash도 이 서버와 일치한다.

## 이어서 읽을 정리 문서

- [설계·풀링 비교 인수인계](../../docs/COMPACT_MODEL_DESIGN_HANDOFF.md)
- [기존 증강 종류·설정·모델별 결과](../../docs/AUGMENTATION_HISTORY.md)
- [연속 화자프로파일·마스킹·환경 증강 계획](../../docs/CONTINUOUS_SPEAKER_AUGMENTATION_PLAN.md)

위 정리 문서의 당시 실행중 상태는 과거 기록이다. 주파수18조건은 현재 완료됐으며 새 연속 증강 계획은 실행하지 않았다.

[전체 비교표](all18_comparison.csv) · [예측 재계산 검사](all18_audit.json) · [팀원 복원 검사](self2_import_audit.json)
