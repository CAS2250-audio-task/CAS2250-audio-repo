"""Compact report handoff; reads metrics only, never audio/models or credentials."""
import json,os
from pathlib import Path

def read(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return {}

def roots():
    seen=set();found=[]
    for base in [Path.cwd(),Path.home(),Path('/workspace'),Path('/work'),Path('/data')]:
        if not base.is_dir():continue
        for directory,dirs,_ in os.walk(base):
            p=Path(directory)
            if p in seen:dirs[:]=[];continue
            seen.add(p)
            if (p/'runs/final-four-gpu-2225').is_dir():found.append(p)
            dirs[:]=[n for n in dirs if not n.startswith('.') and n not in ['datasets','runs','reports','venv','env','node_modules','site-packages','__pycache__']] if len(p.relative_to(base).parts)<6 else []
    return found

def brief(row):
    c=row.get('candidate',{})
    return dict(id=c.get('id'),budget=c.get('budget'),dev_F1_pct=100*row['dev_f1'] if 'dev_f1' in row else None,noise_FP=row.get('noise_fp'),original_F1_pct=100*row['original_f1'] if 'original_f1' in row else None)

all_results=[]
for root in roots():
    for role in range(4):
        run=root/'runs/final-four-gpu-2225';nextstage=read(run/f'nextstage_role{role}_status.json')
        if not nextstage:continue
        folder=root/'reports/final-selected-test'/f'role{role}'
        final=read(folder/'final_status.json');winner=read(folder/'winner.json')
        scores=read(folder/'development_scores.json')
        protocol=read(folder/'selection_protocol.json')
        def key(c):return (c.get('mode'),c.get('condition'),c.get('checkpoint_sha256'),(c.get('detector') or {}).get('sha256'))
        observed={key(r.get('candidate',{})):r for r in scores.get('scores',[])}
        expanded=[]
        for c in protocol.get('candidates',[]):
            if not c.get('eligible'):continue
            match=observed.get(key(c))
            if match:expanded.append(dict(match,candidate=c))
        if expanded:scores['scores']=expanded
        report=dict(root=str(root),role=role,training=nextstage.get('stage'),evaluation=final.get('stage',read(run/f'finaleval_role{role}_status.json').get('stage','NOT_QUEUED')),error=final.get('error'),winner=brief(winner),selection_complete=winner.get('selection_complete'),cases=[brief(r) for r in scores.get('scores',[]) if r.get('candidate',{}).get('eligible')],test=[])
        test=read(folder/'S01_test/metrics.json')
        for r in test.get('results',[]):
            m=r.get('metrics',{})
            f=r.get('command10_f1',m.get('command10_f1'))
            report['test'].append(dict(condition=r.get('condition'),F1_pct=100*f if f is not None else None,noise_FP=r.get('noise_fp_files'),latency=r.get('latency'),evaluated_files=r.get('noisy_files'),unavailable_files=r.get('unavailable_background_files')))
        report['tail_and_LS']=[{k:s.get(k) for k in ['label','completed_epochs','selected_best_epoch','noisy_F1_pct','original_F1_pct','noise_only_FPR_pct','train_speakers','validation_speakers']} for s in nextstage.get('summaries',[])]
        all_results.append(report)
print('REPORT_SUMMARY_BEGIN')
print(json.dumps(all_results,ensure_ascii=False,separators=(',',':')))
print('REPORT_SUMMARY_END')
