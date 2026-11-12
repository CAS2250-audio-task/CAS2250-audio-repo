# seed42: ref 조건의 사전학습 비교

같은203개 검증 파일에서 음성 사전학습만 정답 10개, ImageNet만 정답 22개다.

## 음성 사전학습만 맞힌 파일

| 정답 | 파일 | ImageNet 예측 | 음성 사전학습 예측 |
| --- | --- | --- | --- |
| 복귀 | [spk05_033.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_033.wav) | 정지 | 복귀 |
| 오른쪽 | [spk05_066.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_066.wav) | 왼쪽 | 오른쪽 |
| 오른쪽 | [spk05_081.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_081.wav) | 왼쪽 | 오른쪽 |
| 재시작 | [spk05_105.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_105.wav) | 시작 | 재시작 |
| 재시작 | [spk05_112.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_112.wav) | 시작 | 재시작 |
| 출발 | [spk05_170.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/출발/spk05_170.wav) | 시작 | 출발 |
| 출발 | [spk05_180.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/출발/spk05_180.wav) | 시작 | 출발 |
| 후진 | [spk05_189.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_189.wav) | 시작 | 후진 |
| 후진 | [spk05_191.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_191.wav) | 시작 | 후진 |
| 후진 | [spk05_195.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_195.wav) | 오른쪽 | 후진 |

## ImageNet만 맞힌 파일

| 정답 | 파일 | ImageNet 예측 | 음성 사전학습 예측 |
| --- | --- | --- | --- |
| 대기 | [spk05_004.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/대기/spk05_004.wav) | 대기 | 재시작 |
| 대기 | [spk05_020.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/대기/spk05_020.wav) | 대기 | 재시작 |
| 복귀 | [spk05_034.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_034.wav) | 복귀 | 왼쪽 |
| 복귀 | [spk05_035.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_035.wav) | 복귀 | 정지 |
| 복귀 | [spk05_043.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/복귀/spk05_043.wav) | 복귀 | 왼쪽 |
| 시작 | [spk05_051.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_051.wav) | 시작 | 재시작 |
| 시작 | [spk05_055.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_055.wav) | 시작 | 재시작 |
| 시작 | [spk05_063.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/시작/spk05_063.wav) | 시작 | 재시작 |
| 오른쪽 | [spk05_079.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_079.wav) | 오른쪽 | 왼쪽 |
| 오른쪽 | [spk05_080.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/오른쪽/spk05_080.wav) | 오른쪽 | 후진 |
| 왼쪽 | [spk05_090.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/왼쪽/spk05_090.wav) | 왼쪽 | 오른쪽 |
| 왼쪽 | [spk05_095.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/왼쪽/spk05_095.wav) | 왼쪽 | silence |
| 왼쪽 | [spk05_098.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/왼쪽/spk05_098.wav) | 왼쪽 | 대기 |
| 재시작 | [spk05_113.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/재시작/spk05_113.wav) | 재시작 | 시작 |
| 전진 | [spk05_125.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_125.wav) | 전진 | 후진 |
| 전진 | [spk05_127.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_127.wav) | 전진 | 대기 |
| 전진 | [spk05_133.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/전진/spk05_133.wav) | 전진 | 후진 |
| 정지 | [spk05_162.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/정지/spk05_162.wav) | 정지 | 전진 |
| 후진 | [spk05_198.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_198.wav) | 후진 | 정지 |
| 후진 | [spk05_201.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_201.wav) | 후진 | 전진 |
| 후진 | [spk05_202.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_202.wav) | 후진 | 전진 |
| 후진 | [spk05_203.wav](/root/audio/.worktrees/pretrain/datasets/task2/fixed-1p5-v1/audio/후진/spk05_203.wav) | 후진 | 대기 |
