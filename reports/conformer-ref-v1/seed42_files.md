# seed42: ImageNet와 한국어 Conformer 예측 차이

각 실행의 검증 F1 최고 checkpoint를 같은203개 파일에서 비교했다.

| 파일 | 정답 | ImageNet | Conformer 고정 | Conformer 미세조정 |
| --- | --- | --- | --- | --- |
| [spk05_032.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_032.wav) | 복귀 | 시작 | 복귀 | 복귀 |
| [spk05_033.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_033.wav) | 복귀 | 정지 | 복귀 | 복귀 |
| [spk05_038.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_038.wav) | 복귀 | 후진 | 복귀 | 복귀 |
| [spk05_042.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_042.wav) | 복귀 | 대기 | 복귀 | 복귀 |
| [spk05_059.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_059.wav) | 시작 | 재시작 | 재시작 | 시작 |
| [spk05_060.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_060.wav) | 시작 | 시작 | 재시작 | 시작 |
| [spk05_062.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_062.wav) | 시작 | 시작 | 재시작 | 시작 |
| [spk05_066.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_066.wav) | 오른쪽 | 왼쪽 | 오른쪽 | 오른쪽 |
| [spk05_081.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_081.wav) | 오른쪽 | 왼쪽 | 오른쪽 | 오른쪽 |
| [spk05_082.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_082.wav) | 오른쪽 | 왼쪽 | 오른쪽 | 오른쪽 |
| [spk05_105.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_105.wav) | 재시작 | 시작 | 재시작 | 재시작 |
| [spk05_107.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_107.wav) | 재시작 | 시작 | 재시작 | 재시작 |
| [spk05_112.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_112.wav) | 재시작 | 시작 | 재시작 | 재시작 |
| [spk05_130.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_130.wav) | 전진 | 대기 | 전진 | 전진 |
| [spk05_170.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/출발/spk05_170.wav) | 출발 | 시작 | 출발 | 출발 |
| [spk05_180.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/출발/spk05_180.wav) | 출발 | 시작 | 출발 | 출발 |
| [spk05_189.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_189.wav) | 후진 | 시작 | 후진 | 후진 |
| [spk05_191.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_191.wav) | 후진 | 시작 | 후진 | 후진 |
| [spk05_195.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_195.wav) | 후진 | 오른쪽 | 후진 | 후진 |
