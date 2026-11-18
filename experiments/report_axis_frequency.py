"""Audit the completed experiment-C follow-up and write the matched comparison."""
import json
import math
from pathlib import Path
import shlex

import numpy as np
import pandas as pd
import torch

from experiments.low_frequency_grid.trainer import ROOT, sha, write
from experiments.low_frequency_grid.models import models as build_models

REPORT=ROOT/'reports/axis-frequency-followup'
RUNS={
    'baseline':'lowfreq-frequency-standard-spk03-e1000',
    'A':'lowfreq-axis_frequency-standard-spk03-e1000',
    'B':'lowfreq-axis_frequency_attn-standard-spk03-e1000',
}
ARCHITECTURES={'baseline':'frequency','A':'axis_frequency','B':'axis_frequency_attn'}


def macro_f1(frame):
    cm=np.bincount(frame.true_idx.to_numpy()*11+frame.pred_idx.to_numpy(),minlength=121).reshape(11,11)
    denom=cm.sum(0)+cm.sum(1)
    return float(np.divide(2.*cm.diagonal(),denom,out=np.zeros(11),where=denom!=0).mean())


def inspect_run(key,name):
    folder=ROOT/'runs'/name/'spk03'
    protocol_path=ROOT/'reports'/name/'protocol.json'
    protocol=json.loads(protocol_path.read_text())
    summary=json.loads((ROOT/'reports'/name/'summary.json').read_text())[0]
    history=json.loads((folder/'history.json').read_text())
    assert len(history)==500 and [r['epoch'] for r in history]==list(range(1,501))
    assert all(r['model']=='cnn_pooled_query' for r in history)
    for row in history:
        lr=1e-5+(.001-1e-5)*(1+math.cos(math.pi*(row['epoch']-1)/999))/2
        assert row['lr']==lr
    selected=max(history,key=lambda row:row['zero_f1'])
    checkpoint=folder/'cnn_pooled_query_best.pt'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    last=torch.load(folder/'last.pt',map_location='cpu',weights_only=True)
    model=build_models(42,ARCHITECTURES[key])['cnn_pooled_query']
    model.load_state_dict(state['model'],strict=True)
    assert sum(p.numel() for p in model.parameters())==summary['parameters']
    assert summary['completed_epochs']==last['epoch']==500
    assert summary['planned_epochs']==protocol['epochs']==1000
    assert selected['epoch']==state['epoch']==summary['best_epoch']
    assert selected['zero_f1']==state['score']['zero_f1']
    assert state['protocol_sha256']==last['protocol_sha256']==sha(protocol_path)
    assert last['best']['cnn_pooled_query']['epoch']==state['epoch']
    prediction_path=folder/'cnn_pooled_query_best_predictions.csv.gz'
    predictions=pd.read_csv(prediction_path)
    assert len(predictions)==203*25 and not predictions.path.str.contains('spk01').any()
    assert predictions.path.str.contains('spk03').all()
    groups=list(predictions.groupby(['kind','noise_id','draw']))
    assert len(groups)==25
    original=predictions[predictions.kind=='original']
    assert len(original)==203
    expected_paths=set(original.path)
    assert len(expected_paths)==203
    zero=[]
    for (kind,noise,draw),group in groups:
        assert len(group)==203 and set(group.path)==expected_paths
        assert sorted(group.true_idx.unique())==list(range(11))
        if kind=='mixed_0db':
            assert draw in (0,1,2)
            zero.append(macro_f1(group))
    assert len(zero)==24 and predictions[predictions.kind=='mixed_0db'].noise_id.nunique()==8
    recomputed_original=macro_f1(original)
    recomputed_zero=float(np.mean(zero))
    assert abs(recomputed_original-summary['best']['original_f1'])<1e-6
    assert abs(recomputed_zero-summary['best']['zero_f1'])<1e-6
    final=json.loads((folder/'cnn_pooled_query_final_metrics.json').read_text())
    assert final['summary']==summary and len(final['best_units'])==25
    assert abs(np.mean([r['macro_f1'] for r in final['best_units'] if r['kind']=='mixed_0db'])-recomputed_zero)<1e-6
    count=8*3*20
    false_positives=round(summary['noise_only_fpr']*count)
    assert false_positives/count==summary['noise_only_fpr']
    row=dict(model=key,architecture=ARCHITECTURES[key],completed_epochs=500,
        best_epoch=summary['best_epoch'],original_f1=summary['best']['original_f1'],
        zero_f1=summary['best']['zero_f1'],noise_only_fpr=summary['noise_only_fpr'],
        parameters=summary['parameters'],training_only_projection_parameters=summary['training_only_projection_parameters'],
        noise_only_false_positives=false_positives,noise_only_count=count,
        checkpoint=str(checkpoint.relative_to(ROOT)))
    evidence=dict(history_rows=len(history),original_predictions=203,mixed_predictions=203*24,
        noise_draw_units=24,predictions_recomputed_original_f1=recomputed_original,
        predictions_recomputed_zero_f1=recomputed_zero,
        checkpoint_sha256=sha(checkpoint),last_checkpoint_sha256=sha(folder/'last.pt'),
        protocol_sha256=sha(protocol_path),predictions_sha256=sha(prediction_path),
        earliest_strict_maximum_verified=True,learning_rate_schedule_verified=True,
        checkpoint_strict_load_verified=True,parameter_count_verified=True)
    return row,protocol,evidence,state


def command(key):
    return shlex.join(['.venv/bin/python','-u','-m','experiments.low_frequency_grid.run',
        '--architecture',ARCHITECTURES[key],'--frontend','standard','--folds','spk03',
        '--objective','paired_supcon','--positive-mode','same_command','--supcon-weight','0.1',
        '--temperature','0.1','--epochs','1000','--schedule-epochs','1000','--stop-after','500',
        '--output','runs/'+RUNS[key]])


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    rows={};protocols={};evidence={};states={}
    for key,name in RUNS.items():
        rows[key],protocols[key],evidence[key],states[key]=inspect_run(key,name)
    # Match the actual settings, data and prepared noise hashes, not experiment labels.
    fields=('seed','folds','epochs','split','input','frontend','objective','augmentation',
            'environment','optimizer','monitor','selection','final','labels','sources','audio_sha256',
            'test_audio_read','test_evaluated')
    for key in ('A','B'):
        assert all(protocols[key][field]==protocols['baseline'][field] for field in fields)
        assert protocols[key]['architecture_ablation']['frontend']==protocols['baseline']['architecture_ablation']['frontend']
        assert torch.equal(states[key]['mean'],states['baseline']['mean'])
        assert torch.equal(states[key]['std'],states['baseline']['std'])
        assert all(sha(ROOT/path)==digest for path,digest in protocols[key]['source_sha256'].items())
    assert rows['B']['parameters']-rows['A']['parameters']==2225
    assert rows['A']['parameters']-rows['baseline']['parameters']==168
    preflight=json.loads((REPORT/'preflight.json').read_text())
    assert preflight['status']=='passed'
    assert all(sha(ROOT/path)==digest for path,digest in preflight['source_sha256'].items())
    for result,key in zip(preflight['results'],('A','B')):
        assert result['parameters']==rows[key]['parameters'] and result['all_gradients_finite']
    trace_rows=[]
    for epoch in range(1,501):
        hashes={key:sha(ROOT/'runs'/name/'spk03'/f'augmentation_{epoch:03d}.json') for key,name in RUNS.items()}
        assert len(set(hashes.values()))==1, f'Augmentation mismatch at epoch {epoch}'
        trace_rows.append(dict(epoch=epoch,**hashes))
    pd.DataFrame(trace_rows).to_csv(REPORT/'augmentation_trace_hashes.csv',index=False)
    deltas={}
    for label,left,right in [('B_minus_A','B','A'),('A_minus_baseline','A','baseline')]:
        deltas[label]={metric+'_pp':100*(rows[left][metric]-rows[right][metric])
                       for metric in ('original_f1','zero_f1','noise_only_fpr')}
        deltas[label]['parameters']=rows[left]['parameters']-rows[right]['parameters']
    pd.DataFrame(rows.values()).to_csv(REPORT/'comparison.csv',index=False)
    write(REPORT/'comparison.json',dict(rows=list(rows.values()),deltas=deltas))
    write(REPORT/'commands.json',{key:command(key) for key in ('A','B')})
    write(REPORT/'completion_audit.json',dict(status='passed',matched_protocol_fields=list(fields),
        completed_epochs_per_model=500,augmentation_epochs_identical=500,
        normalization_bitwise_equal=True,current_source_hashes_verified=True,
        preflight_sha256=sha(REPORT/'preflight.json'),runs=evidence,
        report_source_sha256=sha(Path(__file__))))
    lines=['# 실험 C 후속: 축 분리 전처리 + 두 구역 주파수 CNN','',
        'A와 B 모두 실제 500 epoch 학습과 선택 체크포인트의 최종 평가를 완료했다. 기존 500 epoch 두 구역 CNN과도 조건을 대조했다.','',
        '- A: 기본 Mel → 8채널 축 분리 잔차 전처리(5×1 → 1×3) → 두 구역 주파수 CNN → 시간 Pooled → 11클래스.',
        '- B: A의 전처리 직후, CNN 직전에 기존 FrequencyAttention 1블록을 추가.',
        '- Frontend와 BandRoutes는 quick_frequency40 구현을 재사용한다. CNN 본체는 기존 FrequencyTimeCNN의 주파수·시간 분리 블록이며, 중간 48채널과 최종 128채널 각각의 저·고 대역을 평균/연결해 32차원으로 투영하고 더한다.',
        '- 전처리와 세로 어텐션의 잔차 출력은 0으로 초기화했다. CNN·Pooled·분류기·SupCon projector 초기값을 기존 대조군과 맞췄다.','',
        'D1 fixed-1p5-v1, 16 kHz/1.5초, spk02·04·05 학습 609개 / spk03 검증 203개, seed 42. 기본 64 log10 Mel(20–7800 Hz, FFT512/win400/hop160, 151프레임), 원본 학습 오디오만으로 대역별 정규화를 적합했다.','',
        '손실은 paired CE + 0.1 SupCon, 온도 0.1, 같은 명령어를 positive로 사용하고 silence는 SupCon에서 제외했다. 포먼트 증강 p=0.5 뒤 학습 소음 6종 중 하나를 p=1.0, SNR U[-5,20] dB로 혼합했다. 속도·피치·SpecAugment는 사용하지 않았다.','',
        'AdamW(lr=0.001, weight decay=0.0001), 원본 기준 배치 32(참조/잡음 쌍 64개), gradient clip=5. 1000 epoch cosine(lr 최저 1e-5)의 첫 500 epoch를 실행했다. 500 epoch에 cosine을 끝까지 내리는 일정이 아니다. 매 epoch 소음 8종의 고정 draw0 0 dB 전체 11클래스 macro F1 평균이 엄격하게 증가할 때 체크포인트를 저장하여 동률이면 가장 이른 epoch를 선택했다.','',
        '| 모델 | 선택 epoch | 원본 macro F1 | 0 dB 전체 macro F1 | 잡음 단독 오반응률 | 추론 파라미터 |',
        '|---|---:|---:|---:|---:|---:|']
    for key in ('baseline','A','B'):
        r=rows[key]
        label='기존 두 구역 CNN' if key=='baseline' else key
        lines.append(f"| {label} | {r['best_epoch']} | {100*r['original_f1']:.2f}% | {100*r['zero_f1']:.2f}% | {100*r['noise_only_fpr']:.2f}% ({r['noise_only_false_positives']}/480) | {r['parameters']:,} |")
    lines+=['','0 dB 지표는 소음 8종 × 3구간 각각의 11클래스 macro F1을 구한 뒤 24개를 평균했다. 잡음 단독 평가는 같은 소음 8종 × 3구간 × 20개 = 480개에서 silence 이외를 출력한 비율이다. 학습 전용 SupCon projector 4,192개 파라미터는 추론 파라미터 수에서 제외했다.','',
        '| 차이 (%p) | 원본 macro F1 | 0 dB 전체 macro F1 | 잡음 단독 오반응률 | 파라미터 차이 |',
        '|---|---:|---:|---:|---:|']
    for label,description in [('B_minus_A','B − A'),('A_minus_baseline','A − 기존 두 구역 CNN')]:
        d=deltas[label]
        lines.append(f"| {description} | {d['original_f1_pp']:+.2f} | {d['zero_f1_pp']:+.2f} | {d['noise_only_fpr_pp']:+.2f} | {d['parameters']:+,} |")
    lines+=['','F1은 높을수록, 잡음 오반응률은 낮을수록 좋다. 한 검증 화자·한 seed 결과이며, 선택에 사용한 draw0가 최종 3구간 평가에도 포함된다. 독립 test 결과나 여러 seed에 대한 우열을 뜻하지 않는다.','',
        '샘플 배치 forward/backward, 모든 gradient의 유한성, 잔차 전처리·중간 경로·세로 어텐션의 학습 신호를 확인했다. 저장 예측 5,075개/모델에서 F1을 다시 계산했고, 각 500개 학습 이력·선택 기준·학습률·체크포인트 해시·정규화·소음/오디오 해시를 검증했다. A/B/기존 대조군의 500 epoch 증강 trace는 모두 동일하다. 자세한 근거는 preflight.json, completion_audit.json, augmentation_trace_hashes.csv에 보존했다.','',
        '## 실행 명령','',
        '```bash','bash experiments/run_axis_frequency_followup.sh','```','',
        '개별 학습 명령:','',
        '```bash',command('A'),command('B'),'```','',
        '두 구성은 기존 experiments.low_frequency_grid.run 진입점의 --architecture 선택지로 사용할 수 있다. 프런트엔드/평가/손실/체크포인트 선택은 기존 trainer를 그대로 사용한다. 재실행 시 일치하는 protocol의 마지막 optimizer 체크포인트에서 이어서 실행한다. 의존성은 저장소 .venv에 soundfile==0.13.1, praat-parselmouth==0.4.6을 설치했고, 기존 torch/torchaudio 2.3.1+cu121과 RTX 3090을 사용했다.','',
        '## 저장 경로','',
        '- 결과표: `reports/axis-frequency-followup/comparison.csv` 및 `comparison.json`.',
        '- 실행 명령: `reports/axis-frequency-followup/commands.json`.',
        '- 학습 로그: `runs/axis-frequency-followup/axis_frequency.log`, `axis_frequency_attn.log`.']
    for key in ('A','B'):
        lines.append(f"- {key} 선택 체크포인트: `{rows[key]['checkpoint']}`.")
        lines.append(f"- {key} 마지막 epoch500 + optimizer: `runs/{RUNS[key]}/spk03/last.pt`.")
        lines.append(f"- {key} 전체 이력·증강 trace·최종 예측/지표: `runs/{RUNS[key]}/spk03/`; 고정 protocol: `reports/{RUNS[key]}/protocol.json`.")
    lines.append(f"- 기존 비교 기록: `reports/{RUNS['baseline']}/summary.json`.")
    (REPORT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(rows=list(rows.values()),deltas=deltas,audit='passed'),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
