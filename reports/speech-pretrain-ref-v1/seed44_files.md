# seed44: ref 조건의 사전학습 비교

같은203개 검증 파일에서 음성 사전학습만 정답 22개, ImageNet만 정답 22개다.

## 음성 사전학습만 맞힌 파일

| 정답 | 파일 | ImageNet 예측 | 음성 사전학습 예측 |
| --- | --- | --- | --- |
| 시작 | [spk05_047.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_047.wav) | 재시작 | 시작 |
| 시작 | [spk05_063.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_063.wav) | 재시작 | 시작 |
| 오른쪽 | [spk05_065.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_065.wav) | 왼쪽 | 오른쪽 |
| 오른쪽 | [spk05_069.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_069.wav) | 왼쪽 | 오른쪽 |
| 오른쪽 | [spk05_073.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_073.wav) | 왼쪽 | 오른쪽 |
| 오른쪽 | [spk05_077.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_077.wav) | 왼쪽 | 오른쪽 |
| 오른쪽 | [spk05_083.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_083.wav) | 왼쪽 | 오른쪽 |
| 전진 | [spk05_128.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_128.wav) | 정지 | 전진 |
| 전진 | [spk05_132.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_132.wav) | 후진 | 전진 |
| 전진 | [spk05_135.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_135.wav) | 정지 | 전진 |
| 전진 | [spk05_142.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_142.wav) | 복귀 | 전진 |
| 정지 | [spk05_145.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/정지/spk05_145.wav) | 대기 | 정지 |
| 정지 | [spk05_146.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/정지/spk05_146.wav) | 대기 | 정지 |
| 정지 | [spk05_147.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/정지/spk05_147.wav) | 대기 | 정지 |
| 정지 | [spk05_152.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/정지/spk05_152.wav) | 전진 | 정지 |
| 정지 | [spk05_155.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/정지/spk05_155.wav) | 대기 | 정지 |
| 정지 | [spk05_161.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/정지/spk05_161.wav) | 대기 | 정지 |
| 출발 | [spk05_181.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/출발/spk05_181.wav) | 왼쪽 | 출발 |
| 후진 | [spk05_189.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_189.wav) | 대기 | 후진 |
| 후진 | [spk05_192.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_192.wav) | 대기 | 후진 |
| 후진 | [spk05_193.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_193.wav) | 대기 | 후진 |
| 후진 | [spk05_195.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_195.wav) | 대기 | 후진 |

## ImageNet만 맞힌 파일

| 정답 | 파일 | ImageNet 예측 | 음성 사전학습 예측 |
| --- | --- | --- | --- |
| 대기 | [spk05_004.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/대기/spk05_004.wav) | 대기 | 재시작 |
| 복귀 | [spk05_025.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_025.wav) | 복귀 | 정지 |
| 복귀 | [spk05_032.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_032.wav) | 복귀 | 정지 |
| 복귀 | [spk05_034.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_034.wav) | 복귀 | 왼쪽 |
| 복귀 | [spk05_038.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_038.wav) | 복귀 | 시작 |
| 복귀 | [spk05_042.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_042.wav) | 복귀 | 시작 |
| 복귀 | [spk05_043.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_043.wav) | 복귀 | 왼쪽 |
| 오른쪽 | [spk05_066.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_066.wav) | 오른쪽 | 왼쪽 |
| 오른쪽 | [spk05_067.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_067.wav) | 오른쪽 | 왼쪽 |
| 오른쪽 | [spk05_070.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_070.wav) | 오른쪽 | 왼쪽 |
| 오른쪽 | [spk05_071.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_071.wav) | 오른쪽 | 왼쪽 |
| 오른쪽 | [spk05_072.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_072.wav) | 오른쪽 | 왼쪽 |
| 오른쪽 | [spk05_079.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_079.wav) | 오른쪽 | 왼쪽 |
| 왼쪽 | [spk05_091.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/왼쪽/spk05_091.wav) | 왼쪽 | 오른쪽 |
| 재시작 | [spk05_107.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_107.wav) | 재시작 | 시작 |
| 재시작 | [spk05_113.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_113.wav) | 재시작 | 시작 |
| 재시작 | [spk05_115.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_115.wav) | 재시작 | 시작 |
| 재시작 | [spk05_120.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_120.wav) | 재시작 | 시작 |
| 전진 | [spk05_133.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_133.wav) | 전진 | 후진 |
| 전진 | [spk05_141.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_141.wav) | 전진 | 재시작 |
| 정지 | [spk05_156.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/정지/spk05_156.wav) | 정지 | 전진 |
| 후진 | [spk05_198.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_198.wav) | 후진 | 시작 |
