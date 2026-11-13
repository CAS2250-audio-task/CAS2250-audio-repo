# 실험 브랜치와 결과 안내

최신 코드·완료 결과·청취 샘플을 한곳에서 보려면 **`attention-pitch-v1`** 브랜치를 사용한다.
기존 ResNet 실험은 `study/jaehun`에, Conformer 실험은 아래 단계별 브랜치에 보존한다.

| 원격 브랜치 | 내용 | 주요 보고서 |
|---|---|---|
| [study/jaehun](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/study/jaehun) | 기존 ResNet·보조음향 특징 CV 및 오류 분석 | `reports/auxiliary-cv-v1/README.md` |
| [pretrain](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/pretrain) | 한국어 Conformer 사전학습, 소음 혼합·잡음제거 비교·학습 연장 | `reports/conformer-*/`, `reports/denoise-eval-v1/` |
| [overnight-regularization-formant-aux](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/overnight-regularization-formant-aux) | WD·Dropout·증강·포먼트·보조특징 및 청취 샘플 | `reports/overnight-v1/README.md` |
| [attention-pitch-v1](https://github.com/CAS2250-audio-task/CAS2250-audio-repo/tree/attention-pitch-v1) | 위 Conformer 실험 이력 + VAD attention·국소 피치·경량화 + 새 소음 평가 | 아래 링크 |

## 최신 완료 결과

- [마스킹·피치·경량화 비교](reports/attention-pitch-v1/README.md): 기존 소음8종의 0dB 4-fold F1은 기준91.2974%, attention91.6418%, pitch90.8364%, 결합91.3023%, 8블록88.9261%, 6블록85.3750%.
- 관측 최고는 attention이다. 사전 채택 규칙인 평균 +0.5%p를 통과하지 못해 공식 선택은 기준 유지. 8/6블록은 이 기준 조건에서 축소했다.
- [새 소음 평가](reports/attention-unseen-noise-v1/README.md): attention checkpoint를 고정한 채 필터·재학습 없이 새 소음7파일/4유형을 평가했다. 유형별 동일가중 F1 76.9992%. 기존8종91.6418%를 재현했다.
- 두 소음 집합은 난이도가 다르므로 평균 차이가 모델 개선·퇴보량을 의미하지 않는다. 이 역시 개발 화자02~05의 검증이며 test01 평가가 아니다.
- [새 피치 청취](reports/attention-pitch-v1/listening/README.md), [포먼트·피치 다양화 청취](reports/overnight-v1/listening/pitch-formant-diversity/README.md), [어려운 소음 사례](reports/overnight-v1/listening/difficult-noises/README.md).
- [상세 대화 인계](CHAT_HANDOFF_ATTENTION_PITCH.md): 최근 학습 실험의 조건·구조·결과와 로컬 경로. 이후 추가된 새 소음 평가는 위 별도 보고서를 함께 읽는다.

## 저장 범위

Git에 보존하는 것은 실험 코드, 고정 조건·검증 기록, 결과 CSV/압축 예측, 학습곡선 및 청취 샘플이다.
`.gitignore`의 `runs/` 아래 대용량 모델 체크포인트, 실행 환경, 전체 오디오 캐시와 데이터 일부는 서버 로컬에 남아 있다.
따라서 저장소 clone만으로 훈련된 모델 파일까지 내려받는 것은 아니다. 체크포인트 위치와 SHA256은 결과의 fold JSON/protocol에 기록돼 있다.

로컬 최신 worktree: `/root/audio/.worktrees/attention-pitch`.
새 소음 평가까지 모두 완료된 상태를 보존하며, 이 업로드에서 새 학습은 수행하지 않았다.
