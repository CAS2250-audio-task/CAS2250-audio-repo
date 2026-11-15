# 파라미터·추론 지연 비교

같은 세션에서 학습을 잠깐 중지하고 측정한 뒤 자동 재개했다. RTX3090, CPU2threads,batch1,같은1.5초음성,각20회워밍업 후200회 측정. 아래는중앙값이다.

| 구조 | 모델 | 추론파라미터 | GPU모델만 | GPU전처리+전송+모델 | CPU전처리+모델 |
|---|---|---:|---:|---:|---:|
| baseline | cnn_mean | 77,835 | 1.00ms | 1.32ms | 2.29ms |
| baseline | cnn_pooled_query | 78,859 | 1.15ms | 1.45ms | 2.37ms |
| baseline | cnn_self_cls | 86,315 | 1.54ms | 1.86ms | 2.80ms |
| short | cnn_mean | 76,555 | 1.02ms | 1.34ms | 2.22ms |
| short | cnn_pooled_query | 77,579 | 1.14ms | 1.46ms | 2.31ms |
| short | cnn_self_cls | 85,035 | 1.55ms | 1.87ms | 2.72ms |
| local | cnn_mean | 79,915 | 1.08ms | 1.40ms | 2.89ms |
| local | cnn_pooled_query | 80,939 | 1.21ms | 1.52ms | 2.94ms |
| local | cnn_self_cls | 88,395 | 1.61ms | 1.93ms | 3.48ms |
| both | cnn_mean | 78,635 | 1.07ms | 1.40ms | 2.86ms |
| both | cnn_pooled_query | 79,659 | 1.28ms | 1.52ms | 2.89ms |
| both | cnn_self_cls | 87,115 | 1.62ms | 1.94ms | 3.42ms |

baseline=기존1.105초CNN. short=0.625초범위축소. local=0.245초중간특징잔차연결. both=둘다. readout입력은모두32차원이다.

중간연결은2080개파라미터를더하고,범위축소는1280개를줄인다. 둘다는순800개증가다. SupCon투영부4192개는학습에만사용해추론파라미터에서제외했다.

두고정된1.5초입력을이미확보한뒤의처리시간이며,WAV해독·모델로딩·음성수집·발화종료판단·통신은포함하지않는다. 최대수용영역과음성수집지연을모델실행ms와혼동하지않는다.

측정당시both는학습중이었다. 위시간은해당체크포인트의고정구조시간으로최종인식성능을의미하지않는다. 학습가중치는달라도각구조는정적dense연산을수행한다.

[전체수치·p95](latency_comparison.csv) · [환경·측정범위·재개기록](latency_comparison.json)
