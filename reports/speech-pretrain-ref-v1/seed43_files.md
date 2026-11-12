# seed43: ref 조건의 사전학습 비교

같은203개 검증 파일에서 음성 사전학습만 정답 12개, ImageNet만 정답 22개다.

## 음성 사전학습만 맞힌 파일

| 정답 | 파일 | ImageNet 예측 | 음성 사전학습 예측 |
| --- | --- | --- | --- |
| 시작 | [spk05_048.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_048.wav) | 재시작 | 시작 |
| 시작 | [spk05_063.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_063.wav) | 재시작 | 시작 |
| 왼쪽 | [spk05_085.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/왼쪽/spk05_085.wav) | 오른쪽 | 왼쪽 |
| 왼쪽 | [spk05_090.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/왼쪽/spk05_090.wav) | 오른쪽 | 왼쪽 |
| 왼쪽 | [spk05_095.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/왼쪽/spk05_095.wav) | 복귀 | 왼쪽 |
| 재시작 | [spk05_115.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_115.wav) | 대기 | 재시작 |
| 전진 | [spk05_141.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_141.wav) | 정지 | 전진 |
| 전진 | [spk05_142.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_142.wav) | 출발 | 전진 |
| 출발 | [spk05_164.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/출발/spk05_164.wav) | 시작 | 출발 |
| 출발 | [spk05_174.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/출발/spk05_174.wav) | 재시작 | 출발 |
| 출발 | [spk05_180.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/출발/spk05_180.wav) | 시작 | 출발 |
| 출발 | [spk05_181.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/출발/spk05_181.wav) | 재시작 | 출발 |

## ImageNet만 맞힌 파일

| 정답 | 파일 | ImageNet 예측 | 음성 사전학습 예측 |
| --- | --- | --- | --- |
| 대기 | [spk05_006.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/대기/spk05_006.wav) | 대기 | 재시작 |
| 대기 | [spk05_012.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/대기/spk05_012.wav) | 대기 | 정지 |
| 대기 | [spk05_015.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/대기/spk05_015.wav) | 대기 | 정지 |
| 대기 | [spk05_020.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/대기/spk05_020.wav) | 대기 | 정지 |
| 복귀 | [spk05_025.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_025.wav) | 복귀 | 정지 |
| 복귀 | [spk05_026.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_026.wav) | 복귀 | 정지 |
| 복귀 | [spk05_034.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_034.wav) | 복귀 | 왼쪽 |
| 복귀 | [spk05_035.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_035.wav) | 복귀 | 정지 |
| 복귀 | [spk05_036.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_036.wav) | 복귀 | 정지 |
| 복귀 | [spk05_038.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_038.wav) | 복귀 | 시작 |
| 시작 | [spk05_055.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_055.wav) | 시작 | 왼쪽 |
| 오른쪽 | [spk05_079.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_079.wav) | 오른쪽 | 왼쪽 |
| 오른쪽 | [spk05_080.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_080.wav) | 오른쪽 | 시작 |
| 오른쪽 | [spk05_082.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_082.wav) | 오른쪽 | 전진 |
| 오른쪽 | [spk05_083.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_083.wav) | 오른쪽 | 왼쪽 |
| 왼쪽 | [spk05_098.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/왼쪽/spk05_098.wav) | 왼쪽 | 대기 |
| 재시작 | [spk05_105.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_105.wav) | 재시작 | 시작 |
| 재시작 | [spk05_113.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_113.wav) | 재시작 | 시작 |
| 재시작 | [spk05_120.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_120.wav) | 재시작 | 시작 |
| 후진 | [spk05_199.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_199.wav) | 후진 | 전진 |
| 후진 | [spk05_201.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_201.wav) | 후진 | 전진 |
| 후진 | [spk05_202.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_202.wav) | 후진 | 전진 |
