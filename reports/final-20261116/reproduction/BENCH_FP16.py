"""Frozen-candidate S02 FP32/FP16 classifier benchmark. No training or S01 access.

Autocast leaves the saved weights and parameter count unchanged. Detector,
mel/Burg, crop, scalers and decoding remain FP32/CPU as in the reference path.
"""
import argparse,contextlib,hashlib,importlib.util,json,os,sys,time
from datetime import datetime,timezone,timedelta
from pathlib import Path

KST=timezone(timedelta(hours=9))

def read(path):
    return json.loads(Path(path).read_text())

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.bench.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def candidate_from(data):
    if 'candidate' in data:return data['candidate']
    if isinstance(data.get('winner'),dict):return data['winner'].get('candidate',data['winner'])
    return data

def gpu_jobs(gpu,own_pid=None,proc_root=Path('/proc')):
    """Inspect process identity/device only; never expose environment contents."""
    jobs=[];own_pid=os.getpid() if own_pid is None else own_pid
    modules={'study.train','study.train_off','study.train_smoothing','experiments.detector_final.train'}
    files={'pipeline_eval.py','pipeline_eval_extended.py','EVAL_AMBIENT_TEST.py','EVAL_SNR_TEST.py','BENCH_FP16.py'}
    for path in proc_root.glob('[0-9]*/cmdline'):
        try:
            pid=int(path.parent.name)
            if pid==own_pid:continue
            command=[v for v in path.read_bytes().decode().split('\0') if v]
            if not any(v in modules or Path(v).name in files for v in command):continue
            values=(path.parent/'environ').read_bytes().split(b'\0')
            device=next((v.split(b'=',1)[1].decode() for v in values if v.startswith(b'CUDA_VISIBLE_DEVICES=')),None)
            if device is None or device==str(gpu):jobs.append(dict(pid=pid,program=next((Path(v).name for v in command if v in modules or Path(v).name in files),'unknown')))
        except (OSError,UnicodeError,ValueError):continue
    return jobs

def metrics_by_group(rows,pred,metric):
    import numpy as np
    groups={}
    for noise in range(8):
        ids=[i for i,r in enumerate(rows) if r['view']=='noisy' and r.get('noise_id')==f'noise{noise}']
        assert len(ids)==200
        groups[f'noise{noise}']=metric([rows[i]['label'] for i in ids],[pred[i] for i in ids])
    original=[i for i,r in enumerate(rows) if r['view']=='reference']
    negative=[i for i,r in enumerate(rows) if r['view']=='noise_only']
    assert len(original)==200 and len(negative)==80
    return dict(zero_command10_f1=float(np.mean([v['command10_f1'] for v in groups.values()])),
        original=metric([rows[i]['label'] for i in original],[pred[i] for i in original]),
        noise_fp_files=sum(pred[i]!=10 for i in negative),noise_only_files=80,by_noise=groups)

def adoption(metrics32,metrics16,pred32,pred16,latency,fp16_error=None):
    if fp16_error:return dict(adopt=False,precision='fp32',reason='FP16 unsupported or failed: '+fp16_error)
    if pred16 is None or pred32!=pred16:return dict(adopt=False,precision='fp32',reason='Development predictions changed')
    if not latency.get('measured'):return dict(adopt=False,precision='fp32',reason='No uncontended resident latency measurement')
    baseline=latency['fp32'];mixed=latency['fp16']
    faster=mixed['p50_ms']<=.9*baseline['p50_ms'];tail=mixed['p95_ms']<=1.1*baseline['p95_ms']
    return dict(adopt=bool(faster and tail),precision='fp16_autocast_classifier_only' if faster and tail else 'fp32',
        reason='Exact development prediction parity and latency thresholds passed' if faster and tail else 'Less than10% median speedup or p95 worsened more than10%',
        prediction_parity_required=True,minimum_median_speedup_pct=10,maximum_p95_regression_pct=10)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-root',type=Path,required=True);parser.add_argument('--candidate-json',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--detector-checkpoint',type=Path);parser.add_argument('--gpu',default='0')
    parser.add_argument('--adapter',type=Path,default=Path(__file__).with_name('pipeline_eval_extended.py'))
    parser.add_argument('--deadline',default='2026-11-16T23:45:00+09:00')
    parser.add_argument('--samples',type=int,default=30);parser.add_argument('--warmup',type=int,default=5)
    args=parser.parse_args()
    assert args.samples>=10 and args.warmup>=1
    deadline=datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:deadline=deadline.replace(tzinfo=KST)
    end=deadline.timestamp()
    def check():
        if time.time()>=end:raise TimeoutError('Optimization deadline reached; no adoption')
    check();raw=args.raw_root.resolve();work=raw/'runs/final-classifier-2215/classifier-study-20261116'
    candidate=candidate_from(read(args.candidate_json));condition=candidate['condition'];mode=candidate['mode']
    assert condition in ['mean','mean_aux'] and mode in ['actual','off']
    checkpoint=args.checkpoint or Path(candidate.get('checkpoint',candidate.get('classifier_checkpoint','')))
    detector_info=candidate.get('detector');detector_path=args.detector_checkpoint
    if mode=='actual' and detector_path is None:
        detector_path=Path(detector_info['path']) if isinstance(detector_info,dict) else Path(candidate['detector_checkpoint'])
    assert checkpoint.is_file()
    if mode=='actual':assert detector_path and detector_path.is_file()
    metadata=read(work/'bank/metadata.json')
    validation=[r for r in metadata['rows'] if r['split']=='validation']
    assert len(validation)==1880
    assert {r['speaker'] for r in validation if r['label']!=10}=={'S02'},'Only S02 development is permitted'
    protocol=dict(version=1,candidate=candidate,checkpoint_sha256=sha(checkpoint),
        detector_sha256=sha(detector_path) if mode=='actual' else None,bank_sha256=sha(work/'bank/metadata.json'),
        adapter_sha256=sha(args.adapter),script_sha256=sha(__file__),source_waveforms='exact original3480-bank S02 validation subset1880',
        precisions=['fp32','CUDA autocast float16 CLASSIFIER ONLY'],detector_precision='unchanged FP32',
        preprocessing='same CPU mel/Burg, same checkpoint scalers, same crop/decode/EOF; features cached once for parity only',
        samples=args.samples,warmup=args.warmup,latency_order='interleaved FP32/FP16, reversed every sample',
        adoption_rule='all1880 development predictions exact; residentp50>=10%faster; residentp95<=10%worse; idleGPU required',
        S01_audio_opened=False,S01_used_for_selection=False,training=False,deadline_kst=deadline.astimezone(KST).isoformat())
    args.output.mkdir(parents=True,exist_ok=True)
    frozen=args.output/'benchmark_protocol.json'
    if frozen.exists():assert read(frozen)==protocol,'Existing benchmark uses a different frozen protocol'
    else:write(frozen,protocol)
    status=args.output/'status.json'
    write(status,dict(stage='initializing',S01_audio_opened=False))
    os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',CLASSIFIER_WORK=str(work))
    sys.path.insert(0,str(work))
    import numpy as np
    import torch
    from final_shared import reconstruct
    spec=importlib.util.spec_from_file_location('benchmark_adapter',args.adapter)
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(42)
    state=torch.load(checkpoint,map_location='cuda',weights_only=False)
    auxiliary=condition=='mean_aux'
    model=adapter.make_model('basic','mean',auxiliary,state.get('aux_scaler'),42).cuda().eval()
    model.load_state_dict(state['model'],strict=True)
    detector=adapter.CropDetector(detector_path) if mode=='actual' else None
    frontend=adapter.Features('standard');scaler=state['mel_scaler']
    rows,waves,_=reconstruct(raw,metadata,True)
    assert len(rows)==1880
    @torch.inference_mode()
    def encode(wave):
        crop=detector.crop(wave) if detector else wave
        return None if crop is None else frontend.encode(crop,scaler,auxiliary=auxiliary)
    def amp(precision):
        return torch.autocast(device_type='cuda',dtype=torch.float16) if precision=='fp16' else contextlib.nullcontext()
    @torch.inference_mode()
    def predict(items,precision,finite=False):
        output=[10]*len(items);valid=[i for i,x in enumerate(items) if x is not None]
        if valid:
            batch,lengths,_=adapter.collate([items[i] for i in valid],'cuda')
            with amp(precision):logits=model(batch,lengths)
            if finite:assert torch.isfinite(logits).all().item(),'Nonfinite model output'
            predictions=logits.argmax(1).cpu().tolist()
            for i,y in zip(valid,predictions):output[i]=y
        return output
    check();write(status,dict(stage='cache_development_features',cases=1880))
    encoded=[]
    for i,wave in enumerate(waves):
        check();encoded.append(encode(wave))
        if i%200==0:write(status,dict(stage='cache_development_features',completed=i,total=1880))
    predictions={};scores={};error=None
    for precision in ['fp32','fp16']:
        write(status,dict(stage='prediction_parity',precision=precision))
        try:
            values=[]
            for start in range(0,1880,64):check();values.extend(predict(encoded[start:start+64],precision,finite=True))
            predictions[precision]=values;scores[precision]=metrics_by_group(rows,values,adapter.metrics)
        except Exception as exc:
            if precision=='fp32':raise
            error=repr(exc);break
    write(args.output/'development_metrics.json',dict(metrics=scores,fp16_error=error))
    if 'fp16' in predictions:
        differences=[i for i,(a,b) in enumerate(zip(predictions['fp32'],predictions['fp16'])) if a!=b]
        write(args.output/'prediction_comparison.json',[dict(source_id=r['source_id'],view=r['view'],noise_id=r.get('noise_id'),truth=r['label'],fp32=a,fp16=b) for r,a,b in zip(rows,predictions['fp32'],predictions['fp16'])])
    else:differences=None
    latency=dict(measured=False,reason='FP16 failed' if error else 'not yet measured')
    indices=np.linspace(0,1879,args.samples,dtype=int).tolist()
    busy=gpu_jobs(args.gpu)
    if not error and not busy:
        times={'fp32':[],'fp16':[]};gpu_times={'fp32':[],'fp16':[]}
        try:
            write(status,dict(stage='latency_benchmark',samples=args.samples))
            for precision in ['fp32','fp16']:
                for i in range(args.warmup):check();predict([encode(waves[indices[i%len(indices)]])],precision)
            for iteration,index in enumerate(indices):
                check()
                if gpu_jobs(args.gpu):raise RuntimeError('Other GPU work started; discard timing')
                order=['fp32','fp16'] if iteration%2==0 else ['fp16','fp32']
                for precision in order:
                    torch.cuda.synchronize();begin=time.perf_counter()
                    predict([encode(waves[index])],precision)
                    torch.cuda.synchronize();times[precision].append((time.perf_counter()-begin)*1000)
            # Separate cached-feature GPU-only timing; it is not the whole pipeline.
            for iteration,index in enumerate(indices):
                check()
                if gpu_jobs(args.gpu):raise RuntimeError('Other GPU work started; discard timing')
                if encoded[index] is None:continue
                batch,lengths,_=adapter.collate([encoded[index]],'cuda')
                for precision in (['fp32','fp16'] if iteration%2==0 else ['fp16','fp32']):
                    start=torch.cuda.Event(enable_timing=True);finish=torch.cuda.Event(enable_timing=True)
                    with torch.inference_mode():
                        start.record()
                        with amp(precision):model(batch,lengths).argmax(1)
                        finish.record();finish.synchronize()
                    gpu_times[precision].append(start.elapsed_time(finish))
            latency=dict(measured=True,samples=args.samples,warmup=args.warmup,device=torch.cuda.get_device_name(),
                includes='resident waveform; unchanged detector+crop/gate, CPU mel/Burg, H2D, classifier+argmax,D2H',
                excludes='file IO/modelload/audio collection waiting',lookahead_ms=80 if detector else 0,
                detector_frontend_lookahead_ms=12.5 if detector else 0,
                **{precision:dict(p50_ms=float(np.median(times[precision])),p95_ms=float(np.percentile(times[precision],95)),
                    classifier_gpu_p50_ms=float(np.median(gpu_times[precision])) if gpu_times[precision] else None) for precision in times})
        except Exception as exc:latency=dict(measured=False,reason=repr(exc))
    elif busy:latency=dict(measured=False,reason='Other GPU training/inference present',other_jobs=busy)
    result=dict(stage='complete',condition=condition,mode=mode,selected_checkpoint_epoch=state.get('epoch'),
        metrics=scores,fp16_error=error,development_cases=1880,prediction_disagreements=len(differences) if differences is not None else None,
        latency=latency,adoption=adoption(scores.get('fp32'),scores.get('fp16'),predictions.get('fp32'),predictions.get('fp16'),latency,error),
        classifier_parameters=sum(p.numel() for p in model.parameters()),
        detector_parameters=sum(p.numel() for p in detector.model.parameters()) if detector else 0,
        classifier_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        checkpoint_bytes=checkpoint.stat().st_size,checkpoint_unchanged_sha256=sha(checkpoint),
        checkpoint_storage_reduced=False,parameter_count_reduced=False,
        precision_note='Autocast keeps saved FP32 parameters/checkpoint unchanged; no pruning or weight conversion',
        framework=torch.__version__,cuda_version=torch.version.cuda,cudnn_version=torch.backends.cudnn.version(),
        tf32_allowed=False,cpu_threads=2,
        S01_audio_opened=False,S01_used_for_selection=False,training=False,protocol_sha256=sha(frozen))
    write(args.output/'benchmark.json',result);write(status,result)
    print('FP16_BENCH_COMPLETE',json.dumps({k:result[k] for k in ['condition','mode','prediction_disagreements','adoption','classifier_parameters','detector_parameters']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        # Failed benchmarks never modify or adopt a precision mode.
        try:
            if '--output' in sys.argv:
                destination=Path(sys.argv[sys.argv.index('--output')+1])
                write(destination/'status.json',dict(stage='failed',error=repr(exc),adoption=dict(adopt=False,precision='fp32'),S01_audio_opened=False,training=False))
        except Exception:pass
        raise
