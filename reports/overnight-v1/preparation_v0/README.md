# Overnight: 정규화 → 포먼트 증강 → 보조음향 → DFN3

상태: `preparing`, 조건 ``, 화자 ``, epoch ``.

별도 브랜치 overnight-regularization-formant-aux. 원본 test01은 사용하지 않는다.
공통 epoch10 LAST에서 모델·AdamW·RNG를 복원하고100까지 학습한다. 초기10epoch는 이전실험 공통 상태이며 새 후보 변화는epoch11부터 적용한다.
4fold·seed42·batch16·609draw/epoch. 인코더1e-5/head5e-4 유지, 전체 인코더 학습, 조기 종료 없음.
WD1e-3과encoder dropout0.2를 각각 비교, 둘 다 유효하면 결합. 선택 조건에서 소음+포먼트 약/넓은 변형을 비교한다.
포먼트는 고정2변형/파일, 피치중앙값·범위·길이 유지. train에서만 p0.5, 기존noise p0.5와독립적이다. 변형 후 실제 혼합파형에서 보조 특징을 추출한다.
보조 필수 세트1·2·3·7, 시간 여유에4·5·6. 원본train 유효프레임으로만 scaler를 적합한다. aux CNN을 시간 정렬해인코더 출력에잔차로더하며256→11head유지, 초기aux projection은0이다.
조건 선택은 전체3draw0dB 개발4fold F1이 기준보다0.5%p 이상 높고 최저화자F1하락1%p이내일때. 미세한변화는 통계적우월성으로주장하지않는다.
마지막에는 선택된동일모델에DFN3off/on. 필터후파형에서보조도다시추출한다. 기존 공식B16필터캐시를재사용한다.
저장: 각후보best와전체epoch기록보존. 완료분기의큰LAST optimizer는검증후삭제하고미완료분기만재개상태유지. 기존실험파일은수정하지않는다.

[실시간 상태](progress.json) · [로그](../../runs/overnight-v1/execution.log) · [프로토콜](protocol.json)
