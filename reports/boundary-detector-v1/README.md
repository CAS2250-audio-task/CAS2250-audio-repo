# 검출기 준비 상태

실제새팀데이터는아직받지않았다. 합성자료만으로기능검사했으며정확한경계추출성능이나기존VAD대비우월성은검증하지않았다.

구현: CSV열/단위매핑→원본JSONL,검증,7,137/22,465파라미터causalTCN,프레임BCE학습,원본전체소음혼합,train-only표준화,최고/마지막checkpoint와재개,0ms기본margin의구간추출CLI. 시간별결과와원본/소음조건별지표를보존한다.

검사:causality/모델개수/gradient,짧은파일과padding loss제외,샘플좌표변환/다른화자파일선제필터,잘못된경계거부,발화구간기준SNR,확인지연후경계backdating. 합성2epoch CPU학습·재개·추론과64채널CUDA2epoch훈련완료. CUDA bitwise결정성은별도환경설정이필요하며보장하지않는다.

[사용법과한계](../../experiments/boundary_detector/README.md) · [검사기록](synthetic_checks.json)
