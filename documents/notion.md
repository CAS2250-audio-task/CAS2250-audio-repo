10/16 1차 레포트, 11/13 2차 레포트. 

## audio task의 장점 (다른 task와 비교)

data & labeling → train/validation/test set → 퓨리에 → resnet18

데이터 라벨링을 어떻게 해야하는지에 관한 문제:,

set을 정하는 문제,

퓨리에 등 공학-수학적 지식,

tokenize & computer vision,

가능하면 transformer나 lm과 연계까지, 다른 과제의 문제들을 거의 포괄한다고 생각.

## 적용할 수 있는 개선안

#### data

- freq bin 개수, 시간 간격, spectogram의 해상도, 허수 위상,
- data normalization
    - 데이터를 만들 때, filter를 어떻게 넣을 것인지
        
        ex) 음성인식이면 사용자 특징을 줄여야 하고, 사용자 인식이면 음절의 특징을 줄여야 하나? 그러기 위한 필터는?
        
    - spoil을 막기 위해, 어떤 장치를 해야하나? 장소에 따른 배경음을 없앤다던지, 다른 맥락을 지워야 할 수 잇음..

### 진행 파이프라인 (5인)

- **목표:** keyword spotting을 단순 분류가 아니라, 다른 사용자·환경에서도 작동하는 temporal-signal decoder로 설계. target=keyword, nuisance=speaker/room/device/distance/session.
- **1. Task + pilot:** 5명 모두 소량 데이터로 WAV → log-mel spectrogram → ResNet-18 전체 pipeline이 정상 동작하는지 먼저 확인.
- **2. Data + metadata:** 5명 모두 수집에 참여하되 speaker×room×distance×style 조건이 교차되게 수집. speaker↔room 같은 confound를 피하고 speaker/session/room/distance/device/style 기록.
- **3. Split:** random clip split은 baseline, 핵심 평가는 speaker-disjoint. 여유가 있으면 environment/session-disjoint 추가.
- **4. Shared baseline:** 하나의 공통 repo·baseline을 고정한 뒤 실험을 분기.
- **5. 병렬 실험:** ① representation(STFT/mel/window/hop) ② normalization ③ augmentation(noise/reverb/gain) ④ generalization(domain shift) ⑤ systems(latency/streaming) — 5명이 한 축씩 맡되 고정 역할이 아니라 중반 실험 branch 기준.

#### 병렬 실험 상세

응. 여기서 **병렬 실험**은 “5명이 각자 모델 하나씩 만들어서 성능 경쟁”이 아니라, **동일한 baseline 위에서 서로 다른 원인을 하나씩 건드려서 무엇이 성능/일반화를 바꾸는지 분해하는 것**임.

공통 조건은 먼저 고정해야 함. Keyword spotting 기준으로 `10 keywords + silence = 11 classes`, 기본 `log-mel → ResNet-18`, 그리고 화자 기준 train/val/test split을 공통 baseline으로 둔다. 

그 위에서 5개 branch를 이렇게 이해하면 됨.

- **① Representation — “신호를 어떻게 보여줘야 잘 읽는가?”**
    
    같은 WAV라도 spectrogram 만드는 방식이 달라지면 모델이 보는 정보가 달라짐. `mel bin`, `window length`, `hop length` 정도를 바꿔서 비교. 예를 들어 baseline이 128 mel이라면 `64 vs 128`, hop이 10 ms라면 `5/10/20 ms`. 핵심은 그냥 accuracy가 아니라 **시간 해상도 ↔ 주파수 해상도 trade-off가 실제 keyword 구분에 어떤 영향을 주는지** 보는 것. 너무 많은 조합은 하지 말고 2~4개 설정이면 충분.
    
    2번과 3번에 많은 영향이 있을듯, 그냥 성능측정보다는 필요한 정보와 노이즈를 잘 학습할 수 있는 신호 표현을 정해야 함.
    

- **② Normalization — “무엇을 지우고 무엇을 남겨야 하는가?”**
    
    여기서 가장 중요한 질문은 speaker나 recording gain 같은 nuisance를 줄이면서 keyword 정보는 보존할 수 있느냐임. `none / per-clip / dataset-global` 같은 normalization을 비교하고, 가능하면 unseen-speaker 성능을 본다. 만약 training accuracy는 비슷한데 speaker-disjoint test만 개선되면, **화자 특성보다 keyword 자체에 더 집중하게 됐을 가능성**을 말할 수 있음.
    

- **③ Augmentation — “처음 보는 환경에서도 버티게 만들 수 있는가?”**
    
    학습 중 `noise`, `gain`, `reverb` 등을 일부러 추가하고 clean baseline과 비교. 여기서 중요한 건 augmentation 하나하나의 효과. 예를 들어 `noise only`, `reverb only`, `noise+reverb`. 평가할 때는 원래 test뿐 아니라 room/distance별로 보면 좋음. 목적은 성능 뻥튀기가 아니라 **어떤 nuisance에 대한 invariance를 학습했는지 확인하는 것**.
    

- 모델 설계?

- **④ Generalization / Domain shift — “모델이 진짜 keyword를 배웠는가?”**
    
    이건 다른 네 실험의 평가 기준에 가까운 branch임. `random clip split`, `speaker-disjoint`, 가능하면 `room/session-disjoint`를 비교. 같은 모델이 random에서는 95%, unseen speaker에서는 75%라면 그 20%p가 사실상 **distribution shift 비용**임. 이 실험은 모델 자체를 많이 바꿀 필요가 없고, dataset/split 설계를 정확하게 하는 게 핵심. 과제도 같은 group이 여러 split에 섞이지 않도록 요구하고 있음.
    
    성능 측정과 데이터 관리 측면. 가장 중요하지 않을까. 근데 막상 할 일은 별로 없을수도. 또한 모두가 같이 고려해야 할 부분임
    

- **⑤ Systems / Streaming — “실제로 decoder로 쓰려면 얼마나 싸고 빠른가?”**
    
    모델 accuracy만 보지 않고 `audio window 생성 → spectrogram 계산 → inference`까지 latency를 재면 됨. 예를 들면 `1 s / 2 s / 3 s input window`, 또는 overlap `0/50%`를 비교해서 accuracy와 latency를 같이 본다. “3초 입력이면 정확하지만 반응이 느리고, 1초면 빠르지만 성능이 떨어진다” 같은 결과가 나오면 시스템적으로 의미가 있음.
    
    이 주제는 메인으로 잡기엔 무리가 있음.
    

이걸 구조적으로 보면,

$$
			\text{Representation}
\rightarrow
\text{Normalization/Augmentation}
\rightarrow
\text{Model}
\rightarrow
\text{Generalization evaluation}
\rightarrow
\text{System cost}
$$

이고, 5명이 사실상 **파이프라인의 서로 다른 병목을 하나씩 조사하는 것**임.

중요한 건 각 branch가 결과를 이런 식으로 남기는 것임.

> **Hypothesis → 딱 하나의 변경 → 같은 baseline 조건 → 동일 metric → 결과 → 왜 그런지 해석**
> 

예를 들어 representation 담당자가 mel-bin이 128이 좋았다고 해서 그냥 “128이 최고”라고 끝내는 게 아니라,

**“64→128에서 unseen-speaker F1은 +2.1%p였지만 preprocessing latency는 +18%; 256은 추가 이득 없음”**처럼 내놓는 게 좋음.

그리고 5개 중 실제 보고서에서 중심축은 나는 **④ generalization을 메인 질문**으로 두고, ①②③이 “어떻게 generalization을 개선하는가”, ⑤가 “그 개선을 실제 시스템에서 감당할 수 있는가”로 묶는 게 제일 예쁨.

즉 최종 스토리는:

> **모델이 keyword를 외운 것인가, 화자/환경을 외운 것인가? → domain-disjoint 평가 → representation/normalization/augmentation으로 개선 → 그 개선의 latency 비용까지 측정**
> 

이렇게 되면 5명이 따로 논 게 아니라 **하나의 연구 질문을 5방향에서 파고든 프로젝트**가 됨.

#### AI 심화 확장 가능성

지금 방향을 바꾸지 않고도 CV·tokenization·Transformer·representation learning을 자연스럽게 얹을 수 있음. 핵심은 새 모델을 별도 프로젝트처럼 붙이는 게 아니라, **기존 질문인 target 정보와 nuisance 정보의 분리·일반화** 안으로 흡수하는 것.

- **CV 관점 — spectrogram은 이미지와 같으면서도 다름**
    
    ResNet-18은 2D image의 locality/translation bias를 이용하지만 spectrogram의 두 축은 time × frequency라 자연 이미지의 x/y와 의미가 다름. 따라서 이미지 flip/crop 같은 augmentation을 그대로 적용하기보다, time masking/frequency masking이나 시간·주파수 해상도 변경처럼 audio에 맞는 변환을 비교할 수 있음.
    
    → **현재 파이프라인의 ① Representation + ③ Augmentation에 들어감.**
    

- **Tokenization — spectrogram patch를 token sequence로 보기**
    
    F×T spectrogram을 작은 patch로 나누고 각 patch를 embedding해서 p1, p2, ..., pN → e1, e2, ..., eN 형태의 token sequence로 만들 수 있음. NLP의 word token과 완전히 같지는 않지만, positional encoding·CLS token·self-attention을 실제 audio에 연결할 수 있음.
    
    → **① Representation 뒤, Model 직전에 들어감.** log-mel → patch/token embedding → Transformer.
    

- **Transformer / ViT-style 비교**
    
    기존 log-mel → ResNet-18을 baseline으로 두고, 작은 spectrogram Transformer를 추가 비교. CNN의 local pattern bias와 Transformer의 long-range interaction이 keyword 및 background/nuisance 표현에 어떤 차이를 만드는지 볼 수 있음.
    
    → **Shared baseline 이후 Model 단계에 추가되는 선택적 branch.** ④ Generalization에서 같은 split/metric으로 비교.
    

- **Representation analysis**
    
    CNN/Transformer의 embedding을 뽑아 keyword뿐 아니라 speaker/room을 얼마나 잘 예측할 수 있는지 probe하거나 UMAP/t-SNE로 확인. 예를 들어 keyword 성능은 유지되는데 room/speaker probe 성능이 떨어지면 target/nuisance separation이 개선됐다고 해석할 여지가 생김.
    
    → **①~③ 결과를 해석하고 ④ Generalization을 설명하는 분석 단계에 들어감.**
    

- **Pretrained / self-supervised audio representation**
    
    여유가 있으면 handcrafted log-mel + ResNet과 pretrained audio encoder + 작은 classifier를 비교해 **feature engineering vs representation learning**을 볼 수 있음. 프로젝트 규모상 필수보다는 추가 실험 후보.
    
    → **Model 단계의 심화 branch.**
    

- **Discrete audio token / audio-language model**
    
    Waveform을 discrete token으로 바꾸거나 audio-language model까지 연결하는 것도 가능하지만, 구현 볼륨이 커서 이번 프로젝트에서는 **확장 가능성/보고서 논의** 정도가 적절.
    

**현재 파이프라인에 끼워 넣으면**

$$
			\text{WAV}
\rightarrow
\text{Representation(STFT/log-mel)}
\rightarrow
\text{Normalization/Augmentation}
\rightarrow
\begin{cases}
\text{ResNet-18}\\
\text{Patch/Token embedding} \rightarrow \text{Transformer}
\end{cases}
\rightarrow
\text{Generalization evaluation}
\rightarrow
\text{Embedding/nuisance analysis}
$$

즉 **CV는 representation/augmentation**, **tokenization은 representation→model 사이**, **Transformer는 model**, **pretrained representation은 model 대체/비교**, **embedding analysis는 evaluation 이후 해석**에 들어감. 가장 현실적인 심화는 `ResNet baseline 유지 + patch tokenization/작은 Transformer 비교 + nuisance/generalization 분석` 정도.

- **6. 분석:** accuracy + macro-F1 + confusion matrix + speaker/room/distance별 성능 + 최종 ablation.
- **운용:** 초반은 5명 공동(Task/pilot/data/baseline) → 중반은 1인 1실험축 병렬화 → 후반은 공동 error analysis·결론.