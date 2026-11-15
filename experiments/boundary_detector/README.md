# 발화 경계 검출기: 새 데이터 도착 전 준비 코드

전용 브랜치 `experiment/speech-boundary-detector`. 원본에서 명령어 발화의 시작·끝을 찾는 별도 지도학습 모델이다. 실제 새 팀 데이터는 아직 받지 않았고 합성 파형으로만 기능 검사했다. 저장소의 과거 Silero/200ms-margin 라벨이나 기존 팀 음성을 새 정답으로 사용하지 않았다.

## 데이터 계약

학습 코드는 CSV 배열 순서/현재 열 이름과 무관하게 canonical JSONL만 읽는다. CSV 변환은 `prepare.py`와 매핑 JSON이 담당하므로 CSV 형식이 바뀌면 매핑/어댑터만 조정하면 된다. 원본 파일·화자·시작/끝 단위는 명시해야 한다. 경로/화자/원본 ID를 행 순서로 추측하지 않는다.

예시: `configs/boundary_detector/annotations.example.csv`, `csv_mapping.example.json`, `noises.example.json`.

```bash
python -m experiments.boundary_detector.prepare \
  --csv /data/annotations.csv \
  --mapping configs/boundary_detector/csv_mapping.example.json \
  --audio-root /data/originals --speakers speakerA speakerB \
  --output runs/detector-data-v1/manifest.jsonl
python -m experiments.boundary_detector.validate \
  --manifest runs/detector-data-v1/manifest.jsonl --speakers speakerA speakerB
```

`unit`은 `seconds`/`milliseconds`/`samples`. 샘플 좌표는 기본적으로 원본 샘플레이트 기준이며, 다른 레이트 좌표면 `annotation_sample_rate`를 명시한다. 다른 CSV 이름은 mapping의열이름을 바꾼다. 한 원본의 여러 구간은 여러 행 또는 `segments_col`의 JSON 배열 `[[start,end],...]`로 표현한다. 중복·겹친 구간/잘못된 범위는 자동 수정하지 않고 오류로 보고한다. 명시적 negative 행은 `is_negative=true`, 빈 경계다. 양성인데 경계만 비어 있는 행은 거부한다. 자동 VAD 경계는 `automatic_vad` 등으로 표시하며 수동 정답과 구분한다.

canonical 각 행: `source_id`,절대`audio_path`,`speaker`,`native_sample_rate`,`duration_seconds`,`audio_sha256`,`segments`(초),`annotation_source`. 원본이 같은 행들은 하나로 묶인다. 원본 안 짧은 무성 자음/틈도 제공된 구간 안이면 양성으로 유지한다. 원본 그대로인 클립을 자동으로 오류 처리하지 않되, 전체가 양성인 행은 검수 대상으로 표시한다. 실제 오디오/경계 품질을 코드만으로 보증하지 않는다.

WAV/FLAC은soundfile, 기타 형식은 설치된ffprobe/ffmpeg가 있으면 사용한다. 원본은 수정하지 않고16kHz mono로 메모리에서 변환한다. 이번 합성 검사는WAV경로만 확인했다. 새 M4A 등의 코덱/메타데이터는 업로드 후 실제 디코딩 정합성을 확인해야 한다. 파일이1.5초보다 짧아도 허용한다. 배치용패딩은loss에서 제외하며 inference는원본길이를 사용한다.

## 모델과 시간 규칙

- 64 log-mel,FFT512/window400,hop160(10ms) 또는80(5ms),고정train-only band normalization. 파일별 음량/RMS를 같게 만드는 추가 정규화는 하지 않는다.
- 채널32/64 stem + causal depthwise-separable residual4블록,k5,dilation1/2/4/8 + 시간별 binary head. 시간 평균/attention pooling 없음.
- 채널32는7,137파라미터,64는22,465파라미터. 기본64.
- LayerNorm은같은시각의채널끼리만 수행한다. 시간방향전체정규화로미래가누설되지않게한다. CNN은causal이고mel창은center=True라경계주변의짧은future context를사용한다. 실시간stateful API는아직구현하지않았으며현재CLI는파일단위추론이다.
- 제공된시작/끝을그대로학습하고기본추출margin=0ms. 출력간격이5/10ms라는것이실제오차보장은아니다.
- onset/off hysteresis와연속확인시간은설정값이다. 확인을기다려도추출위치는후보시점으로되돌린다. 기본20ms on확인/50ms off확인은추가패딩이아니다. EOF의미확정off dip은유지하는보수정책이며관련경계오차를검증해야한다.

## 학습

```bash
python -m experiments.boundary_detector.train \
  --manifest runs/detector-data-v1/manifest.jsonl \
  --train-speakers speakerA speakerC speakerD --val-speakers speakerB \
  --noise-manifest /data/noises.json \
  --width 64 --hop 160 --epochs 100 --device cuda \
  --output runs/detector-first-v1
```

매핑에서train/val에쓸화자를먼저선택하고그화자들만오디오를연다. train/val화자및원본hash중복은거부한다. 윈도우를만들기전에원본/화자분리한다. 별도test화자는훈련명령에넣지않는다. 같은녹음세션의파생물역시같은fold에묶어야한다.

원본전체에소음을혼합한뒤경계주변/임의윈도우를뽑는다. SNR은표시된발화구간의원본과해당소음구간파워로계산한다. 원본의기존잡음은분리하지못하므로명목추가SNR이다. noise probability기본0.8,SNR[-5,20]dB,일부원본유지. 실제외부소음만으로negative예시도만든다. 합성0파형negative만학습하지않는다. 훈련창에서공동gain[-6,3]dB를주며peak초과시감쇠한다. 실제소음파일에발화가섞였는지와검출목표에맞는라벨은사전에검수해야한다.

노이즈JSON은train/validation 배열에경로나 `{path,original_group}`을적는다. 경로는절대또는노이즈JSON상대경로다. 양쪽동일파일hash/명시된같은원본group은거부한다. 다른파일명/hash여도같은원본에서나온자료면group을표시해야하며group미제공만으로독립성을보증할수없다. 출처·라이선스관리는원본노이즈manifest에함께보존할수있다. 새소음환경확장은아직실행하지않았다.

BCE지도학습,AdamW1e-3/WD1e-4,cosine→1e-5,gradient clip5. 매epoch원본/고정0dB검증에대해프레임가중평균BCE최소checkpoint를선택한다. event F1과경계오차는별도보고한다. 이선택기준의경계정확도는실제검증후재검토할수있다. classifier용대조학습/보조음향/시간가변증강은추가하지않았다.

`last.pt`는매epochoptimizer와정규화/config/history포함. 동일명령에`--resume`추가하면재개한다. 전체epoch예산을바꾸면cosine스케줄도바뀌므로동일실험resume으로허용하지않는다. 원본/manifest/소스/노이즈hash가바뀌면기존run을덮지말고새output을쓴다. `--allow-noise-free`는명시적인무증강대조/smoke전용이다. 이코드는백그라운드학습을자동시작하지않는다.

## 추론·평가

```bash
python -m experiments.boundary_detector.predict \
  --checkpoint runs/detector-first-v1/best.pt \
  --audio /data/new_original.wav --device cpu \
  --output runs/detector-first-prediction
```

segments.json/시간별probabilities.npy/각추출WAV를새폴더에저장한다. 기본margin0ms,추가여유는`--margin-ms`로만명시한다. 원본을덮지않는다. 모델판정이무발화면빈구간목록을출력한다. 실제경계는아직검증되지않았다.

평가:원본/소음별frame/event F1,미검출event/오검출event,IoU≥0.5로매칭된구간의시작·끝MAE/p95,10ms보다늦은시작/빠른끝잘림수를보존한다. MAE는매칭된구간에조건부이므로미검출수와함께읽는다. 자동VAD라벨평가는기존VAD추종성일수있어정밀경계정확도를위해서는수동검수된검증이필요하다. 추출후명령어분류성능비교는아직연결하지않았다.

## 합성 검사

```bash
python -m experiments.boundary_detector.check
```

32/64모델개수·causality·역전파,샘플단위매핑/8kHz소스/짧은파일,다른화자파일필터,패딩lossmask,잘못된경계거부,SNR기준,경계확인후backdating을검사하고합성WAV로2epoch CPU학습/재개/추론을확인한다. 실팀성능검증은아니다. 반복실행은 `--output runs/synthetic-check-v2`처럼 새 출력경로를 지정한다. CUDA bitwise재현성은플랫폼환경에영향을받으며현재warn-only검사라완전일치를보장하지않는다.
