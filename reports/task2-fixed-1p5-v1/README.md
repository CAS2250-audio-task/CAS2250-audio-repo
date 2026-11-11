# 1.5초 전체 데이터 baseline 실행 결과

최종 데이터 `task2-1015-fixed-1p5-v1`의 train 609개로 10 epochs를 학습했습니다.
Validation 203개로 checkpoint를 선택했고 test 203개는 추론하지 않았습니다.

- GPU: NVIDIA GeForce RTX 3090.
- LR: 5e-4, ImageNet pretrained=True, augmentation=False, seed=42.
- Best epoch: 10.
- Validation accuracy: **92.12%**.
- Validation macro-F1: **0.925813**.
- Correct predictions: 187 / 203.

이 값은 validation 성능입니다. 업로드된 70% 노트북은 실행 로그와 split이 없어 직접 비교하지 않습니다.

`history.csv`, `learning_curves.png`에 학습 곡선,
`validation/`에 클래스별 지표·confusion matrix·오분류 기록을 보관했습니다.
`manifest.json`에 실제 코드 커밋, 파일 해시, 환경과 실행 명령이 있습니다.
가중치 checkpoint는 로컬 `runs/task2-fixed-1p5-v1/best.pt`에 있습니다.

```bash
python -m src train --config configs/task2_real.json --output runs/task2-fixed-1p5-v2 --device cuda
```

새 출력 폴더로 재학습하며 기존 run을 덮어쓰지 않습니다.
