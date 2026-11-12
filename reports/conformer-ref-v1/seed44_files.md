# seed44: ImageNet와 한국어 Conformer 예측 차이

각 실행의 검증 F1 최고 checkpoint를 같은203개 파일에서 비교했다.

| 파일 | 정답 | ImageNet | Conformer 고정 | Conformer 미세조정 |
| --- | --- | --- | --- | --- |
| [spk05_042.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_042.wav) | 복귀 | 복귀 | 대기 | 대기 |
| [spk05_047.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_047.wav) | 시작 | 재시작 | 시작 | 시작 |
| [spk05_055.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_055.wav) | 시작 | 재시작 | 대기 | 시작 |
| [spk05_059.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_059.wav) | 시작 | 재시작 | 재시작 | 시작 |
| [spk05_062.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_062.wav) | 시작 | 시작 | 재시작 | 시작 |
| [spk05_063.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_063.wav) | 시작 | 재시작 | 시작 | 시작 |
| [spk05_065.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_065.wav) | 오른쪽 | 왼쪽 | 오른쪽 | 오른쪽 |
| [spk05_069.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_069.wav) | 오른쪽 | 왼쪽 | 오른쪽 | 오른쪽 |
| [spk05_073.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_073.wav) | 오른쪽 | 왼쪽 | 오른쪽 | 오른쪽 |
| [spk05_077.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_077.wav) | 오른쪽 | 왼쪽 | 오른쪽 | 오른쪽 |
| [spk05_081.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_081.wav) | 오른쪽 | 왼쪽 | 오른쪽 | 오른쪽 |
| [spk05_082.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_082.wav) | 오른쪽 | 왼쪽 | 오른쪽 | 오른쪽 |
| [spk05_083.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_083.wav) | 오른쪽 | 왼쪽 | 오른쪽 | 오른쪽 |
| [spk05_113.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_113.wav) | 재시작 | 재시작 | 시작 | 재시작 |
| [spk05_128.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_128.wav) | 전진 | 정지 | 전진 | 전진 |
| [spk05_130.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_130.wav) | 전진 | 대기 | 전진 | 전진 |
| [spk05_132.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_132.wav) | 전진 | 후진 | 전진 | 전진 |
| [spk05_135.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_135.wav) | 전진 | 정지 | 전진 | 전진 |
| [spk05_142.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_142.wav) | 전진 | 복귀 | 전진 | 전진 |
| [spk05_145.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/정지/spk05_145.wav) | 정지 | 대기 | 정지 | 정지 |
| [spk05_146.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/정지/spk05_146.wav) | 정지 | 대기 | 정지 | 정지 |
| [spk05_147.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/정지/spk05_147.wav) | 정지 | 대기 | 정지 | 정지 |
| [spk05_150.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/정지/spk05_150.wav) | 정지 | 대기 | 정지 | 정지 |
| [spk05_152.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/정지/spk05_152.wav) | 정지 | 전진 | 정지 | 정지 |
| [spk05_154.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/정지/spk05_154.wav) | 정지 | 대기 | 정지 | 정지 |
| [spk05_155.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/정지/spk05_155.wav) | 정지 | 대기 | 정지 | 정지 |
| [spk05_161.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/정지/spk05_161.wav) | 정지 | 대기 | 정지 | 정지 |
| [spk05_181.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/출발/spk05_181.wav) | 출발 | 왼쪽 | 출발 | 출발 |
| [spk05_189.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_189.wav) | 후진 | 대기 | 후진 | 후진 |
| [spk05_192.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_192.wav) | 후진 | 대기 | 후진 | 후진 |
| [spk05_193.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_193.wav) | 후진 | 대기 | 후진 | 후진 |
| [spk05_195.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_195.wav) | 후진 | 대기 | 후진 | 후진 |
| [spk05_200.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_200.wav) | 후진 | 재시작 | 후진 | 후진 |
| [spk05_201.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_201.wav) | 후진 | 오른쪽 | 후진 | 후진 |
| [spk05_202.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_202.wav) | 후진 | 오른쪽 | 후진 | 후진 |
