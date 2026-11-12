# seed43: ImageNet와 한국어 Conformer 예측 차이

각 실행의 검증 F1 최고 checkpoint를 같은203개 파일에서 비교했다.

| 파일 | 정답 | ImageNet | Conformer 고정 | Conformer 미세조정 |
| --- | --- | --- | --- | --- |
| [spk05_004.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/대기/spk05_004.wav) | 대기 | 재시작 | 대기 | 대기 |
| [spk05_032.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_032.wav) | 복귀 | 대기 | 복귀 | 복귀 |
| [spk05_042.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_042.wav) | 복귀 | 대기 | 대기 | 복귀 |
| [spk05_043.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_043.wav) | 복귀 | 재시작 | 복귀 | 복귀 |
| [spk05_048.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_048.wav) | 시작 | 재시작 | 시작 | 시작 |
| [spk05_059.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_059.wav) | 시작 | 재시작 | 시작 | 시작 |
| [spk05_063.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_063.wav) | 시작 | 재시작 | 시작 | 시작 |
| [spk05_081.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_081.wav) | 오른쪽 | 왼쪽 | 오른쪽 | 오른쪽 |
| [spk05_085.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/왼쪽/spk05_085.wav) | 왼쪽 | 오른쪽 | 왼쪽 | 왼쪽 |
| [spk05_090.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/왼쪽/spk05_090.wav) | 왼쪽 | 오른쪽 | 왼쪽 | 왼쪽 |
| [spk05_095.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/왼쪽/spk05_095.wav) | 왼쪽 | 복귀 | 왼쪽 | 왼쪽 |
| [spk05_107.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_107.wav) | 재시작 | 시작 | 재시작 | 재시작 |
| [spk05_113.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_113.wav) | 재시작 | 재시작 | 시작 | 재시작 |
| [spk05_115.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_115.wav) | 재시작 | 대기 | 재시작 | 재시작 |
| [spk05_130.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_130.wav) | 전진 | 대기 | 전진 | 전진 |
| [spk05_141.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_141.wav) | 전진 | 정지 | 전진 | 전진 |
| [spk05_142.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_142.wav) | 전진 | 출발 | 전진 | 전진 |
| [spk05_164.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/출발/spk05_164.wav) | 출발 | 시작 | 출발 | 출발 |
| [spk05_174.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/출발/spk05_174.wav) | 출발 | 재시작 | 출발 | 출발 |
| [spk05_180.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/출발/spk05_180.wav) | 출발 | 시작 | 출발 | 출발 |
| [spk05_181.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/출발/spk05_181.wav) | 출발 | 재시작 | 출발 | 출발 |
| [spk05_198.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_198.wav) | 후진 | 대기 | 후진 | 후진 |
| [spk05_200.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_200.wav) | 후진 | 시작 | 후진 | 후진 |
