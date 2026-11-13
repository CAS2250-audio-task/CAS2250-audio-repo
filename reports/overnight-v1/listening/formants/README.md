# 실제 포먼트 변형 듣기 샘플

각 이어듣기는 원본 → 포먼트 낮춤 → 포먼트 높임 순서이며 사이에0.4초 간격이 있다.
이번 B조건에서 사용한 실제 파형 캐시의 flags4(포먼트만)이다. 피치·속도 변형·추가 소음 없이 음색 차이를 비교한다.
샘플은 두 변형의 비율이 서로 반대이고 차이가3% 이상인 출발 파일 중 경로순 첫 사례다. 예측 정오 여부는 선택에 쓰지 않았다.
별도 볼륨 정규화 없이16kHz PCM16으로 저장했다. 변형 전 float32 파형 해시는 캐시와 실제 training trace 모두 대조했다.
화자2는02검증fold의학습에포함되지않는다. 이화자의증강이훈련에쓰인03fold의입력을예시로내보냈다.

## spk02 · 출발

원본 `audio/출발/spk02_051.wav`. 포먼트 비율 **0.9293배 → 1.0773배**.

[원본→낮춤→높임 이어듣기](spk02/comparison.wav)

[원본](spk02/original.wav) · [포먼트 낮춤](spk02/formants_lower.wav) · [포먼트 높임](spk02/formants_higher.wav)

## spk03 · 출발

원본 `audio/출발/spk03_166.wav`. 포먼트 비율 **0.9196배 → 1.0325배**.

[원본→낮춤→높임 이어듣기](spk03/comparison.wav)

[원본](spk03/original.wav) · [포먼트 낮춤](spk03/formants_lower.wav) · [포먼트 높임](spk03/formants_higher.wav)
