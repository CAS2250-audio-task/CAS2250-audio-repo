"""Compare trained readouts under identical idle-GPU conditions; resume own training in finally."""
import atexit
import copy
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from experiments.compact_pooling.run import Frontend
from experiments.local_features.models import variant_models

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/root/CAS2250-compact-contrastive-100')
OUT=ROOT/'reports/compact-local-features'
NAMES=('cnn_mean','cnn_pooled_query','cnn_self_cls')
VARIANTS=('baseline','short','local','both')


def main():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    metadata=pd.read_csv(ROOT/'datasets/task2/fixed-1p5-v1/metadata.csv')
    row=metadata[(metadata.speaker_id=='spk03')&(metadata.label!='silence')].iloc[0]
    wave,sr=sf.read(ROOT/'datasets/task2/fixed-1p5-v1'/row.path,dtype='float32')
    assert sr==16000 and wave.shape==(24000,)
    cpu_wave=torch.from_numpy(wave[None])
    # Capture complete checkpoint files before pausing. Checkpoint writes use atomic rename.
    saved={}
    for variant in VARIANTS:
        folder=BASE/'runs/compact-noise-speaker-spk03-e1000/spk03' if variant=='baseline' else ROOT/f'runs/compact-local-{variant}-spk03-e1000/spk03'
        for name in NAMES:saved[(variant,name)]=torch.load(folder/(name+'_best.pt'),map_location='cpu',weights_only=False)
    status=json.loads((ROOT/'runs/compact-local-features/status.json').read_text())
    paused=[];cancel=Path(f'/tmp/compact-latency-resumed-{os.getpid()}');started=None
    def resume():
        for pid in paused[:]:
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
            paused.remove(pid)
        cancel.touch()
    atexit.register(resume)
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Signal{signum}')
    signal.signal(signal.SIGTERM,interrupted)
    output=[]
    try:
        for key,info in status['variants'].items():
            if info['status']!='running':continue
            pid=info['pid'];cmd=Path(f'/proc/{pid}/cmdline')
            if not cmd.exists():continue
            text=cmd.read_bytes().replace(b'\0',b' ').decode()
            if 'experiments.local_features.run' not in text or str(ROOT) not in text:raise RuntimeError('Unexpected training PID')
            current=Path(f'/proc/{pid}/status').read_text()
            if '\nState:\tT' in current:raise RuntimeError('Training was already stopped; refusing to take ownership of its pause')
            os.kill(pid,signal.SIGSTOP);paused.append(pid)
        started=time.monotonic();paused_pids=paused[:]
        # A separate watchdog resumes training even if this benchmark process is killed.
        watchdog='import os,signal,time;from pathlib import Path\npids='+repr(paused_pids)+'\ncancel=Path('+repr(str(cancel))+')\nfor i in range(240):\n if cancel.exists():break\n time.sleep(.5)\nelse:\n for pid in pids:\n  try:os.kill(pid,signal.SIGCONT)\n  except ProcessLookupError:pass\n'
        subprocess.Popen([sys.executable,'-c',watchdog],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        time.sleep(.5)
        gpu_frontend=Frontend().cuda().eval();cpu_frontend=Frontend().eval()
        gpu_wave=cpu_wave.cuda()
        def measure(fn,cuda):
            for _ in range(20):fn()
            timings=[]
            for _round in range(2):
                for _ in range(100):
                    if cuda:torch.cuda.synchronize()
                    tick=time.perf_counter();fn()
                    if cuda:torch.cuda.synchronize()
                    timings.append((time.perf_counter()-tick)*1000.)
            return float(np.median(timings)),float(np.percentile(timings,95))
        with torch.inference_mode():
            # Warm the GPU clock and common frontend before comparing variants.
            for _ in range(100):gpu_frontend(gpu_wave)
            torch.cuda.synchronize()
            for variant in VARIANTS:
                models=variant_models(42,variant)
                for name in NAMES:
                    checkpoint=saved[(variant,name)]
                    model=models[name].cuda().eval();model.load_state_dict(checkpoint['model'])
                    mean=checkpoint['mean'].cuda();std=checkpoint['std'].cuda()
                    feature=(gpu_frontend(gpu_wave)-mean)/std
                    gm,gp=measure(lambda:model(feature),True)
                    em,ep=measure(lambda:model((gpu_frontend(cpu_wave.cuda())-mean)/std).argmax(-1).cpu(),True)
                    cpu_model=copy.deepcopy(model).cpu().eval();cmn=mean.cpu();cst=std.cpu()
                    cm,cp=measure(lambda:cpu_model((cpu_frontend(cpu_wave)-cmn)/cst).argmax(-1),False)
                    output.append(dict(variant=variant,model=name,parameters=sum(p.numel() for p in model.parameters()),checkpoint_epoch=checkpoint['epoch'],
                        gpu_model_median_ms=gm,gpu_model_p95_ms=gp,gpu_frontend_model_median_ms=em,gpu_frontend_model_p95_ms=ep,
                        cpu_frontend_model_median_ms=cm,cpu_frontend_model_p95_ms=cp))
                    print(variant,name,round(em,3),round(cm,3),flush=True)
                    del model,cpu_model
                del models
        elapsed=time.monotonic()-started
    finally:
        resume()
    result=dict(time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        cpu_threads=2,batch=1,input_seconds=1.5,input_path=row.path,warmup_per_measurement=20,timed_runs=200,
        training_pause_seconds=elapsed,training_resumed=True,paused_training_pids=paused_pids,
        excludes='WAV decoding, model loading,1.5s audio acquisition, endpoint detection,communications',
        gpu_pipeline='Decoded CPU waveform -> GPU mel+normalization+classifier -> CPU class index',
        cpu_pipeline='Decoded CPU waveform -> CPU mel+normalization+classifier',
        note='Same session for all12model architectures; trained selected checkpoint weights. Both variant is still training; timings measure architecture only. SupCon projection4192parameters is training-only and excluded.',
        results=output)
    OUT.mkdir(exist_ok=True,parents=True)
    (OUT/'latency_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    pd.DataFrame(output).to_csv(OUT/'latency_comparison.csv',index=False)
    lines=['# 파라미터·추론 지연 비교','',
       '같은 세션에서 학습을 잠깐 중지하고 측정한 뒤 자동 재개했다. RTX3090, CPU2threads,batch1,같은1.5초음성,각20회워밍업 후200회 측정. 아래는중앙값이다.','',
       '| 구조 | 모델 | 추론파라미터 | GPU모델만 | GPU전처리+전송+모델 | CPU전처리+모델 |','|---|---|---:|---:|---:|---:|']
    for r in output:
        lines.append(f'| {r["variant"]} | {r["model"]} | {r["parameters"]:,} | {r["gpu_model_median_ms"]:.2f}ms | {r["gpu_frontend_model_median_ms"]:.2f}ms | {r["cpu_frontend_model_median_ms"]:.2f}ms |')
    lines+=['',
       'baseline=기존1.105초CNN. short=0.625초범위축소. local=0.245초중간특징잔차연결. both=둘다. readout입력은모두32차원이다.',
       '', '중간연결은2080개파라미터를더하고,범위축소는1280개를줄인다. 둘다는순800개증가다. SupCon투영부4192개는학습에만사용해추론파라미터에서제외했다.',
       '', '두고정된1.5초입력을이미확보한뒤의처리시간이며,WAV해독·모델로딩·음성수집·발화종료판단·통신은포함하지않는다. 최대수용영역과음성수집지연을모델실행ms와혼동하지않는다.',
       '', '측정당시both는학습중이었다. 위시간은해당체크포인트의고정구조시간으로최종인식성능을의미하지않는다. 학습가중치는달라도각구조는정적dense연산을수행한다.',
       '', '[전체수치·p95](latency_comparison.csv) · [환경·측정범위·재개기록](latency_comparison.json)']
    (OUT/'LATENCY.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(training_resumed=True,pause_seconds=elapsed,rows=len(output))),flush=True)


if __name__=='__main__':main()
