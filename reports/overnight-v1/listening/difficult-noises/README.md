# 점수가 낮았던 소음 듣기

현재 선택 모델의 검증0dB F1은 식당78.68%, 압축기83.06%다. 이 파일들은 검증용 소음이다.
소음 단독은 실제 평가 구간에서 시작하는10초를 청취용 볼륨으로 조정했다. 아래0dB 혼합본은 실제 평가 캐시 파형을 PCM16으로 내보냈으며 추가 볼륨 정규화·필터·피치·포먼트 변형은 하지 않았다.
오류 사례는 각 화자에서 원본은 정답이고 두 소음의draw0 혼합본은 모두 오답인 파일 중 경로순 첫 사례다.

## 소음 단독

- [식당 소음만10초](restaurant_noise_10s.wav) · 원본 0.088초부터
- [압축기 소음만10초](compressor_noise_10s.wav) · 원본 865.170초부터

## spk02 · 정답: 대기

[원본](spk02_original.wav) — 예측: 대기

- [식당0dB](spk02_restaurant_0dB.wav) — 예측: **후진** (오답)
- [압축기0dB](spk02_compressor_0dB.wav) — 예측: **후진** (오답)

[원본 → 식당0dB → 압축기0dB 이어듣기](spk02_comparison.wav)

## spk03 · 정답: 대기

[원본](spk03_original.wav) — 예측: 대기

- [식당0dB](spk03_restaurant_0dB.wav) — 예측: **silence** (오답)
- [압축기0dB](spk03_compressor_0dB.wav) — 예측: **silence** (오답)

[원본 → 식당0dB → 압축기0dB 이어듣기](spk03_comparison.wav)
