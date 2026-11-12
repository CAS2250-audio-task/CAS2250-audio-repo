# 추가 검증 기록

- 전체 테스트39개 통과. 새 테스트3개는 개발 화자가 각fold의검증에한번씩만등장하고
  test화자는제외되는지, 훈련라벨대조가검증라벨을바꾸지않는지,
  입력대조가음성/정답중목표요소만바꾸는지확인한다.
- [8개CV checkpoint재추론](cv_checkpoint_replay.json)의1,624개예측은저장된파일별예측과모두일치했다.
  라벨대조군의10epoch2,030개예측지표도재계산했다.
- [개발전체중복검사](development_identity_audit.json):4명812개에서원본SHA·원본파일명·WAV/PCM의
  화자간정확중복0. [개별파일검사](audio_identity_audit.csv)와원본파일해시도일치한다.
- 모델이름/구조, 음성입력, 같은데이터분할에서반복된실험인지함께확인했다.
  이전Whisper-base설명표는실제실험모델과다르다. 실제는CNN3블록+Conformer12층,
  특징256차원, 분류층256→11, 전체32,122,699개파라미터다.

이번재검증은고정된조건의개발결과다. CV는seed42한개이며각fold에서최고epoch를고른다.
최종epoch10으로도Conformer평균F1은99.20%였다. 최종test화자는추론하지않았다.
환경·거리·잡음·기기·발화스타일metadata가없고unknown클래스도없어서,
새환경이나등록외발화에대한실제운영성능을보장하지않는다.
정확해시검사로변형·근사중복을모두배제하거나KsponSpeech원음전체와의중복을확인한것은아니다.

```bash
cd /root/audio/.worktrees/pretrain
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  runs/conformer-env/bin/python -m experiments.conformer_sanity
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  runs/conformer-env/bin/python -m experiments.check_conformer_sanity
```

입력대조·동결라벨대조·Conformer는epoch별재개가가능하다.
완료된fold는건너뛰며ResNet의중단실행은별도이름으로보존한후처음부터다시실행한다.
