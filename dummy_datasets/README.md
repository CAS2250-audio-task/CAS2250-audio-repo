# Dummy audio datasets

Colab의 파일 로딩, spectrogram, ResNet forward/backward를 확인하기 위한 합성 데이터입니다. 실제 성능 비교나 과제 결과로 사용하면 안 됩니다.

- `celebrity_voice`: 한국 이름 30개 × Edge TTS 3개 = 90 WAV
- `keyword_spotting`: 키워드 10개 × Edge TTS 3개 + silence 3개 = 33 WAV
- `dog_breed`: 품종명 10개 × 절차적 bark 3개 = 30 WAV

세 데이터셋 모두 class마다 train/validation/test 파일이 하나씩 있습니다. 실제 프로젝트에서는 split마다 여러 독립 source를 수집하세요.
