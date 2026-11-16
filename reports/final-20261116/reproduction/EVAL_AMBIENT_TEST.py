"""Frozen candidates: per-file observed background mean and p90 stress evaluation.
The p90 condition is a conservative lower-SNR stress, NOT a statistical confidence bound.
No model/scaler fitting; manual boundaries used only to construct evaluation conditions.
"""
import argparse,hashlib,importlib.util,json,os,sys,time,urllib.request
from pathlib import Path
from urllib.parse import quote
import numpy as np
SOURCE='https://raw.githubusercontent.com/CAS2250-audio-task/CAS2250-audio-repo/9b198dcf0e4b188ad89d6189477e9ff1cd0fc32f/'


def background_stats(wave,gold,sr=16000):
    n=len(wave);mask=np.ones(n,dtype=bool)
    for a,b in gold:mask[max(0,round((a-.05)*sr)):min(n,round((b+.05)*sr))]=False
    width=round(.02*sr);frames=[i for i in range(n//width) if mask[i*width:(i+1)*width].all()]
    result=dict(available=len(frames)>=10,background_frames=len(frames),background_samples=int(mask.sum()),
        frame_ms=20,gold_exclusion_margin_ms=50,minimum_background_frames=10,p90_is_statistical_CI=False)
    if len(frames)<10:
        result.update(reason='fewer than10 wholly non-speech20ms frames',ambient_mean_power=None,ambient_p90_power=None,stress_target_power=None)
        return result
    dc=float(np.mean(wave[mask].astype(np.float64)))
    powers=np.array([np.mean((wave[i*width:(i+1)*width].astype(np.float64)-dc)**2) for i in frames])
    mean=float(np.mean(powers));p90=float(np.percentile(powers,90))
    result.update(background_global_dc=dc,ambient_mean_power=mean,ambient_p90_power=p90,stress_target_power=max(mean,p90),
        quiet_mean=mean<=1e-12,definition='mean/p90 of disjoint20ms frame powers after one global non-speech DC subtraction')
    return result


def ambient_mix(wave,noise,targets,seed):
    n=len(wave);rng=np.random.default_rng(seed)
    if len(noise)<n:noise=np.tile(noise,int(np.ceil(n/len(noise))))
    for retry in range(100):
        offset=int(rng.integers(len(noise)-n+1));segment=noise[offset:offset+n].astype(np.float64).copy();segment-=segment.mean()
        power=float(np.mean(segment**2))
        if power>=1e-12 or all(v<=1e-12 for v in targets.values()):break
    else:raise ValueError('No audible deterministic external noise segment')
    result={}
    for label,target in targets.items():
        if target<=1e-12:gain=0.;attenuation=1.;y=wave.copy()
        else:
            gain=float(np.sqrt(target/power));y=wave.astype(np.float64)+gain*segment
            attenuation=min(1.,.99/max(float(np.max(np.abs(y))),1e-12));y=(y*attenuation).astype(np.float32)
        result[label]=(y,dict(noise_start_sample=offset,noise_source_power=power,noise_target_power=float(target),
            added_noise_power=float(gain*gain*power),gain=gain,attenuation=attenuation,retry=retry,zero_add=gain==0))
    return result


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def rng_seed(key):return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')

def discover(explicit=None):
    if explicit:return Path(explicit).expanduser().resolve()
    if os.environ.get('DETECTOR_REPO'):return Path(os.environ['DETECTOR_REPO']).expanduser().resolve()
    seen=set()
    for base in [Path.cwd(),Path.home(),Path('/workspace'),Path('/work'),Path('/data')]:
        if not base.is_dir():continue
        for directory,dirs,_ in os.walk(base):
            p=Path(directory)
            if p in seen:dirs[:]=[];continue
            seen.add(p)
            if (p/'runs/detector-diversity-v2/manifest.jsonl').is_file():return p.resolve()
            dirs[:]=[n for n in dirs if not n.startswith('.') and n not in ['datasets','runs','reports','venv','env','node_modules','site-packages','__pycache__']] if len(p.relative_to(base).parts)<6 else []
    raise RuntimeError('Set --raw-root to prepared detector repository')

def source_path(raw,record,split):
    expected='S02' if split=='dev' else 'S01'
    if record['speaker']!=expected:raise ValueError('Source speaker forbidden for requested split')
    original=Path(record['audio_path'])
    relative='datasets/task2/manual-clipped-v1/'+str(original).split('/datasets/task2/manual-clipped-v1/',1)[1]
    path=original if original.is_file() else raw/relative
    if not path.is_file():
        if split!='test':raise FileNotFoundError(path)
        assert record['speaker']=='S01','Only explicitly requested S01 test files may be fetched'
        path.parent.mkdir(parents=True,exist_ok=True)
        data=urllib.request.urlopen(SOURCE+quote(relative,safe='/'),timeout=45).read()
        assert hashlib.sha256(data).hexdigest()==record['audio_sha256'],record['source_id']
        partial=path.with_name(path.name+'.partial');partial.write_bytes(data);partial.replace(path)
    assert sha(path)==record['audio_sha256'],record['source_id']
    return path

def other_gpu_jobs():
    found=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            pid=int(p.parent.name)
            if pid==os.getpid():continue
            args=[s for s in p.read_bytes().decode().split('\0') if s]
            if any(s in ['study.train','study.train_off','study.train_smoothing','experiments.detector_final.train'] or Path(s).name in ['pipeline_eval.py','pipeline_eval_extended.py','EVAL_SNR_TEST.py'] for s in args):found.append(pid)
        except (OSError,UnicodeError,ValueError):pass
    return found


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--condition',choices=['mean','mean_aux'],required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--detector-checkpoint',type=Path);p.add_argument('--mode',choices=['off','actual'],required=True)
    p.add_argument('--split',choices=['dev','test'],required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--raw-root',type=Path);p.add_argument('--gpu',default='0');p.add_argument('--latency',action='store_true');p.add_argument('--freeze-only',action='store_true')
    a=p.parse_args()
    if a.mode=='actual' and not a.detector_checkpoint:p.error('Actual mode needs --detector-checkpoint')
    raw=discover(a.raw_root);work=raw/'runs/final-classifier-2215/classifier-study-20261116'
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',CLASSIFIER_WORK=str(work));sys.path.insert(0,str(work))
    helper=Path(__file__).with_name('pipeline_eval_extended.py');assert helper.is_file(),helper
    metadata_path=work/'bank/metadata.json';metadata=json.loads(metadata_path.read_text())
    manifest_path=raw/'runs/detector-diversity-v2/manifest.jsonl';speaker='S02' if a.split=='dev' else 'S01'
    records=sorted([json.loads(s) for s in manifest_path.read_text().splitlines() if s.strip()],key=lambda r:r['source_id'])
    records=[r for r in records if r['speaker']==speaker];assert len(records)==200
    noise_hashes=[next(r['synthesis']['noise_sha256'] for r in metadata['rows'] if r['split']=='validation' and r['view']=='noisy' and r['noise_id']==f'noise{i}') for i in range(8)]
    protocol=dict(version=1,split=a.split,evaluation_speaker=speaker,condition=a.condition,mode=a.mode,
        classifier_checkpoint_sha256=sha(a.checkpoint),detector_checkpoint_sha256=sha(a.detector_checkpoint) if a.detector_checkpoint else None,
        script_sha256=sha(__file__),adapter_sha256=sha(helper),source_manifest_sha256=sha(manifest_path),classifier_metadata_sha256=sha(metadata_path),
        conditions=['unmodified original','ambient_mean','background_p90_lower_snr'],
        background_rule=dict(exclude_gold_margin_ms=50,disjoint_frame_ms=20,minimum_wholly_background_frames=10,
            DC='one global mean over all eligible non-speech samples',mean='mean of wholly background frame powers',
            stress='max(mean,p90 frame power); conservative lower-SNR stress, NOT statistical confidence interval',
            unavailable='retain original evaluation; added-noise condition unavailable, explicitly report coverage',
            quiet='if observed mean<=1e-12, both added-noise conditions add exactlyzero; no arbitrary floor boost'),
        noise_hashes=noise_hashes,external_noise_rule='same source/offset for both conditions; gain=sqrt(targetPower/mean(DC-removed external noise^2)); jointpeak attenuation',
        snr_proxy='(annotated original power - observed background mean)/(observed background mean + addednoise power), only if numeratorpositive; not isolated clean speech trueSNR',
        manual_annotations_used_for='evaluation background/noise construction only; never runtime crop or classifier input',
        original_condition='unmodified existing background recording, all200 files retained',negative_policy='same80 historical train-reference0dB noise-only waveforms for both conditions',
        reference_checkpoint_sha256=sha(a.detector_checkpoint or work/'reference_detector.pt'),
        eof_policy='hold16frames80ms',decoder=dict(on=.6,off=.4,on_ms=20,off_ms=50),crop_margin_ms=[50,30],
        scaler_refit=False,training=False,test_used_for_selection=False,test_gate='S01 waveforms only after explicit split=test and protocol freeze',
        selected_sources=[dict(source_id=r['source_id'],sha256=r['audio_sha256']) for r in records])
    a.output.mkdir(parents=True,exist_ok=True);freeze=a.output/'candidate_protocol_freeze.json'
    if freeze.exists():assert json.loads(freeze.read_text())==protocol,'Frozen protocol differs; use another output directory'
    else:write(freeze,protocol)
    print('AMBIENT_PROTOCOL_FROZEN',freeze,'split',a.split,flush=True)
    if a.freeze_only:return
    import torch,soundfile as sf
    from study.prepare import mix as vendor_mix,LABELS
    from final_shared import relocate
    spec=importlib.util.spec_from_file_location('ambient_adapter',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    checkpoint=torch.load(a.checkpoint,map_location='cuda',weights_only=False);auxiliary=a.condition=='mean_aux'
    classifier=m.make_model('basic','mean',auxiliary,checkpoint.get('aux_scaler'),42).cuda().eval();classifier.load_state_dict(checkpoint['model'],strict=True)
    frontend=m.Features('standard');scaler=checkpoint['mel_scaler'];detector=m.CropDetector(a.detector_checkpoint) if a.mode=='actual' else None
    reference=float(torch.load(a.detector_checkpoint or work/'reference_detector.pt',map_location='cpu',weights_only=False)['reference_power']);assert reference>0
    lookup={}
    for entries in json.loads((raw/'runs/detector-diversity-v2/noise.json').read_text()).values():
        for entry in entries:
            path=relocate(raw,entry['path']);lookup[sha(path)]=path
    noises=[]
    for digest in noise_hashes:
        wave,rate=sf.read(lookup[digest],dtype='float32');assert rate==16000 and wave.ndim==1;noises.append(wave)
    originals=[];background=[]
    for record in records:
        path=source_path(raw,record,a.split);wave,rate=sf.read(path,dtype='float32');assert rate==16000 and wave.ndim==1
        stats=background_stats(wave,record['segments']);active=np.zeros(len(wave),bool)
        for lo,hi in record['segments']:active[max(0,round(lo*16000)):min(len(wave),round(hi*16000))]=True
        active_power=float(np.mean(wave[active].astype(np.float64)**2)) if active.any() else None
        proxy=active_power-stats['ambient_mean_power'] if stats['available'] and active_power is not None else None
        stats.update(source_id=record['source_id'],annotated_original_power=active_power,estimated_speech_power_proxy=proxy if proxy is not None and proxy>0 else None)
        originals.append((record,wave,LABELS.index(path.parent.name),stats));background.append(stats)
    write(a.output/'per_file_background.json',background)
    @torch.inference_mode()
    def encode(wave):
        crop=detector.crop(wave) if detector else wave
        return None if crop is None else frontend.encode(crop,scaler,auxiliary=auxiliary)
    @torch.inference_mode()
    def predict(encoded):
        result=[]
        for start in range(0,len(encoded),64):
            batch=encoded[start:start+64];valid=[i for i,x in enumerate(batch) if x is not None];values=[10]*len(batch)
            if valid:
                x,lengths,_=m.collate([batch[i] for i in valid],'cuda');guesses=classifier(x,lengths).argmax(1).cpu().tolist()
                for i,g in zip(valid,guesses):values[i]=g
            result.extend(values)
        return result
    negative_encoded=[];negative_rows=[]
    for i,noise in enumerate(noises):
        for j in range(10):
            rng=np.random.default_rng(42+10000+i*100+j)
            for retry in range(100):
                try:wave,trace=vendor_mix(np.zeros(24000,np.float32),noise,[],rng,0.,reference);break
                except ValueError:
                    if retry==99:raise
            negative_encoded.append(encode(wave));negative_rows.append(dict(source_id=f'negative_{i}_{j}',noise_id=f'noise{i}',noise_sha256=noise_hashes[i],waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),synthesis=trace))
    negative_pred=predict(negative_encoded);noise_fp=sum(v!=10 for v in negative_pred)
    write(a.output/'fixed_negative_predictions.json',[dict(r,prediction=v) for r,v in zip(negative_rows,negative_pred)])
    original_encoded=[encode(w) for _,w,_,_ in originals];original_pred=predict(original_encoded)
    original_metrics=m.metrics([y for _,_,y,_ in originals],original_pred)
    write(a.output/'original_predictions.json',[dict(source_id=r['source_id'],truth=y,prediction=v) for (r,_,y,_),v in zip(originals,original_pred)])
    results=[dict(condition='original_already_contains_background',metrics=original_metrics,source_files=200,noise_fp_files=noise_fp,noise_files=80)]
    write(a.output/'metrics.json',dict(stage='running',protocol_sha256=sha(freeze),results=results))
    coverage=sum(s['available'] for s in background);unavailable=[s['source_id'] for s in background if not s['available']]
    for label in ['ambient_mean','background_p90_lower_snr']:
        encoded=[];cases=[];samples=[]
        for record,wave,truth,stats in originals:
            if not stats['available']:continue
            targets=dict(ambient_mean=stats['ambient_mean_power'],background_p90_lower_snr=stats['stress_target_power'])
            if stats['quiet_mean']:targets=dict(ambient_mean=0.,background_p90_lower_snr=0.)
            for i,noise in enumerate(noises):
                mixed,trace=ambient_mix(wave,noise,targets,rng_seed(f'{record["source_id"]}|{i+1}|42'))[label]
                estimated=stats['estimated_speech_power_proxy'];denominator=stats['ambient_mean_power']+trace['added_noise_power']
                trace['snr_proxy_db']=float(10*np.log10(estimated/denominator)) if estimated and denominator>0 else None
                trace['snr_proxy_is_clean_true_SNR']=False
                encoded.append(encode(mixed));cases.append(dict(source_id=record['source_id'],noise_id=f'noise{i}',truth=truth,noise_sha256=noise_hashes[i],ambient=stats,synthesis=trace,waveform_sha256=hashlib.sha256(mixed.tobytes()).hexdigest()))
                if len(samples)<30 and len(cases)%53==1:samples.append(mixed)
        pred=predict(encoded);by_noise={}
        for i in range(8):
            ids=[j for j,r in enumerate(cases) if r['noise_id']==f'noise{i}'];assert len(ids)==coverage
            by_noise[f'noise{i}']=m.metrics([cases[j]['truth'] for j in ids],[pred[j] for j in ids]) if ids else None
        row=dict(condition=label,command10_f1=float(np.mean([v['command10_f1'] for v in by_noise.values() if v])) if coverage else None,
            by_noise=by_noise,original=original_metrics,noise_fp_files=noise_fp,noise_files=80,source_files_total=200,source_files_available=coverage,
            source_files_unavailable=unavailable,noisy_files=len(cases),forced_silence_files=sum(x is None for x in encoded),
            quiet_zero_add_source_files=sum(s['available'] and s['quiet_mean'] for s in background),
            stress_is_statistical_CI=False,scaler_refit=False)
        other=other_gpu_jobs() if a.latency else []
        if a.latency and not other and samples:
            for wave in samples[:5]:predict([encode(wave)])
            times=[]
            for wave in samples:
                torch.cuda.synchronize();tic=time.perf_counter();predict([encode(wave)]);torch.cuda.synchronize();times.append((time.perf_counter()-tic)*1000)
            row['latency']=dict(resident_pipeline_p50_ms=float(np.median(times)),resident_pipeline_p95_ms=float(np.percentile(times,95)),samples=len(times),device=torch.cuda.get_device_name(),includes='CPU mel+Burg ifaux, actual detector ifenabled, crop/gate,H2D,classifier,D2H',excludes='file IO,model load,audio waiting',lookahead_ms=80 if detector else 0,detector_frontend_lookahead_ms=12.5 if detector else 0)
        else:row['latency']=dict(measured=False,reason='other GPU jobs or no available samples or not requested',other_pids=other)
        results.append(row);write(a.output/(label+'_predictions.json'),[dict(r,prediction=v) for r,v in zip(cases,pred)])
        write(a.output/'metrics.json',dict(stage='running',protocol_sha256=sha(freeze),results=results))
        print('AMBIENT_RESULT',a.split,label,'F1',row['command10_f1'],'available',coverage,'/200','noiseFP',noise_fp,flush=True)
    write(a.output/'metrics.json',dict(stage='complete',protocol_sha256=sha(freeze),results=results,S01_audio_read=a.split=='test',scaler_refit=False,training=False,unavailable_source_files=unavailable))
    print('AMBIENT_EVALUATION_COMPLETE',a.output,flush=True)

if __name__=='__main__':main()
