# 수동 확정 데이터 가져오기

원격 data/manual-clipped-1000-20261116 @ff35dab에서 datasets/task2/manual-clipped-v1 전체를 검출기 브랜치로 가져왔다. 원본·클립각1000개,5화자×200개. 새데이터의학습/검증/test분할은아직정하지않았고학습/모델추론은시작하지않았다. 전체확인은파일/좌표/PCM의정합성검사이며사람청취로경계정확도를재검수한것은아니다.

## 중요한 좌표 규칙

원본은trimmed_originals의지정원본이며최초녹음파일바이트원본과다르다. 둘은README의처리이력을따른다. 일부화자는잡음제거/딸깍음보정사본이며22개는앞뒤정리범위가달라졌다.

검출기양성은 `speech_start_in_trimmed_s`~`speech_end_in_trimmed_s`다. `output_*`는발화밖최대30ms여유가포함된클립범위이므로검출기정답과구분한다. `speech_*_s`는정리전지정원본좌표라trimmed_originals에바로쓰지않는다. BOM있는CSV는utf-8-sig로읽고pair_id로join한다. 행순서를기준으로연결하지않는다.

## 검증

1000행ID/화자/명령어대응,confirmed상태,정리전/후시각변환,발화/출력범위포함관계,2000WAV의16kHz mono PCM16,원본/클립길이를확인했다. 클립1000개모두해당정리원본의PCM구간과시작샘플offset0으로정확히일치했다. 원본중복SHA256그룹0개. 1.5초미만원본458개. 양쪽배경여유가각10ms미만인원본은없고시작쪽10ms미만2개는별도검수할수있다.

CSV와백업JSON은배포validation.json의SHA256과일치했다. designated_source_sha256은정리전지정원본용이므로현재trimmed WAV의해시로대신쓰지않고,canonical manifest에는실제로사용할WAV의SHA256을새로저장했다.

## 경로와 재생성

실제데이터: `datasets/task2/manual-clipped-v1/`
Canonical manifest: `runs/manual-clipped-v1-import/manifest.jsonl`

```bash
python -m experiments.boundary_detector.import_manual \
  --dataset datasets/task2/manual-clipped-v1 \
  --output runs/manual-clipped-v1-import \
  --speakers S01 S02 S03 S04 S05
```

재실행은새output을쓴다. user_confirmed는사용자가확정한구간이라는출처이며독립적인음성학적경계검수보증이아니다. canonical에기존spkXX를추정배정하지않는다. 새SXX와실제화자의대응표는manual_data_audit.json 및import출력speaker_ids.json에있다. 예전spk03검증을계승할경우이름대응을먼저확인해야한다.

모든행이명령어양성이며별도silence클립은없다. 원본발화밖실제배경과새환경소음을negative로쓸수있지만다른사람발화포함여부등은추가검수해야한다. 현재로서는기존소음6개를자동선택하거나새학습을시작하지않았다.

[독립정합성검사](manual_data_audit.json) · [배포원문](../../datasets/task2/manual-clipped-v1/README.md)
