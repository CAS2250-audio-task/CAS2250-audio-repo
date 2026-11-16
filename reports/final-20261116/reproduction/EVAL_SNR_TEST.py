"""Frozen-candidate S02 development / explicitly requested S01 test inference.
No training, scaler fitting, threshold search, or result publication.
"""
import argparse,hashlib,importlib.util,json,os,sys,time,urllib.request
from pathlib import Path
from urllib.parse import quote
import numpy as np

LEVELS=[0.0,5.0,10.0]
SOURCE='https://raw.githubusercontent.com/CAS2250-audio-task/CAS2250-audio-repo/9b198dcf0e4b188ad89d6189477e9ff1cd0fc32f/'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.replace(path)

def rng_seed(key):return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')

def mix_levels(wave,noise,gold,reference,seed,levels,vendor_mix,snr_reference="active"):
    """Choose one audible source/noise offset, then vary gain before peak attenuation."""
    if snr_reference not in ['active','full_file']:raise ValueError('Unknown SNR reference')
    rng=np.random.default_rng(seed)
    n=len(wave)
    if len(noise)<n:noise=np.tile(noise,int(np.ceil(n/len(noise))))
    if snr_reference=='full_file' and gold:
        # File mean uses the entire native waveform; annotations do not set its power.
        power=float(np.mean(wave.astype(np.float64)**2));assert power>0
        for retry in range(100):
            offset=int(rng.integers(len(noise)-n+1))
            segment=noise[offset:offset+n].astype(np.float64).copy();segment-=segment.mean()
            npower=float(np.mean(segment**2))
            if npower>=1e-12:break
        else:raise ValueError('No audible full-window noise crop')
        trace=dict(noise_start_sample=offset,snr_db=0.,snr_reference='full_native_original_waveform_mean_power',
                   gain=float(np.sqrt(power/npower)),attenuation=1.)
        base=None
    else:
        # Noise-only always retains the historical train-reference0dB construction.
        for retry in range(100):
            try:base,trace=vendor_mix(wave,noise,gold,rng,0.,reference);break
            except ValueError:
                if retry==99:raise
        offset=trace['noise_start_sample']
        segment=noise[offset:offset+n].astype(np.float64).copy();segment-=segment.mean()
    rows=[]
    for snr in levels:
        gain=float(trace['gain'])*10**(-float(snr)/20)
        y=wave.astype(np.float64)+gain*segment
        attenuation=min(1.,.99/max(float(np.max(np.abs(y))),1e-12))
        y=(y*attenuation).astype(np.float32)
        if float(snr)==0 and base is not None:assert np.array_equal(y,base),'0dB synthesis changed from vendor'
        rows.append((y,dict(trace,noise_start_sample=offset,snr_db=float(snr),gain=gain,attenuation=attenuation,retry=retry)))
    return rows

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
    p.add_argument('--raw-root',type=Path);p.add_argument('--gpu',default='0');p.add_argument('--latency',action='store_true');p.add_argument('--freeze-only',action='store_true',help='Write/verify protocol without opening any source waveform or using CUDA')
    p.add_argument('--snr-reference',choices=['full_file','active'],default='full_file')
    p.add_argument('--snr-db',nargs='+',type=float,default=LEVELS,help='Explicit levels; default0,5,10. Historical offsets only if explicitly requested.')
    a=p.parse_args()
    if not a.snr_db or not all(np.isfinite(x) for x in a.snr_db):p.error('Finite SNR levels required')
    if a.mode=='actual' and not a.detector_checkpoint:p.error('--mode actual requires --detector-checkpoint')
    raw=discover(a.raw_root);work=raw/'runs/final-classifier-2215/classifier-study-20261116'
    os.environ.update(CUDA_VISIBLE_DEVICES=a.gpu,CUBLAS_WORKSPACE_CONFIG=':4096:8',CLASSIFIER_WORK=str(work))
    sys.path.insert(0,str(work))
    helper=Path(__file__).with_name('pipeline_eval_extended.py')
    if not helper.is_file():raise FileNotFoundError('Place pipeline_eval_extended.py beside this evaluation script')
    metadata_path=work/'bank/metadata.json';metadata=json.loads(metadata_path.read_text())
    manifest_path=raw/'runs/detector-diversity-v2/manifest.jsonl'
    speaker='S02' if a.split=='dev' else 'S01'
    records=[json.loads(s) for s in manifest_path.read_text().splitlines()]
    records=sorted([r for r in records if r['speaker']==speaker],key=lambda r:r['source_id'])
    assert len(records)==200 and {r['speaker'] for r in records}=={speaker}
    noise_hashes=[next(r['synthesis']['noise_sha256'] for r in metadata['rows'] if r['split']=='validation' and r['view']=='noisy' and r['noise_id']==f'noise{i}') for i in range(8)]
    protocol=dict(version=1,split=a.split,evaluation_speaker=speaker,condition=a.condition,mode=a.mode,
        classifier_checkpoint_sha256=sha(a.checkpoint),detector_checkpoint_sha256=sha(a.detector_checkpoint) if a.detector_checkpoint else None,
        classifier_metadata_sha256=sha(metadata_path),source_manifest_sha256=sha(manifest_path),
        reference_checkpoint_sha256=sha(a.detector_checkpoint or work/'reference_detector.pt'),
        script_sha256=sha(__file__),adapter_sha256=sha(helper),levels_db=a.snr_db,snr_reference=a.snr_reference,original_condition='unmodified original recording, already contains background',
        snr_definition=('mean power of each FULLnative original waveform / mean power of the same-length added-noise crop after DC removal' if a.snr_reference=='full_file' else 'annotated original waveform power / added noise power over marked speech samples')+'; original already has background; NOT isolated clean-speech SNR',
        comparability='full_file0dB differs from historical active0dB; do not compare as identical conditions',
        noise_hashes=noise_hashes,noise_offsets='sha256(source_id|noiseIndex+1|42); identical across all gain levels',
        negative_policy='80 identical0dB noise-only waveforms at every level, power fitted on training speakers only',
        eof_policy='hold16frames80ms',decoder=dict(on=.6,off=.4,on_ms=20,off_ms=50),crop_margin_ms=[50,30],
        crop_policy='first detected start to last detected end; missing detection forces silence; OFF uses full native source',
        scaler_refit=False,training=False,test_used_for_selection=False,
        test_gate='S01 waveform access only after explicit --split test and this protocol freeze',
        selected_sources=[dict(source_id=r['source_id'],sha256=r['audio_sha256']) for r in records])
    a.output.mkdir(parents=True,exist_ok=True);freeze=a.output/'candidate_protocol_freeze.json'
    if freeze.exists():assert json.loads(freeze.read_text())==protocol,'Frozen protocol differs; use a distinct output directory'
    else:write(freeze,protocol)
    # Everything above reads metadata/checkpoints hashes only. S01 audio can be opened only below.
    print('CANDIDATE_PROTOCOL_FROZEN',freeze,'split',a.split,flush=True)
    if a.freeze_only:return
    import torch,soundfile as sf
    from study.prepare import mix as vendor_mix,LABELS
    from final_shared import relocate
    spec=importlib.util.spec_from_file_location('snr_adapter',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    torch.set_num_threads(2);torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    checkpoint=torch.load(a.checkpoint,map_location='cuda',weights_only=False)
    auxiliary=a.condition=='mean_aux'
    classifier=m.make_model('basic','mean',auxiliary,checkpoint.get('aux_scaler'),42).cuda().eval();classifier.load_state_dict(checkpoint['model'],strict=True)
    frontend=m.Features('standard');scaler=checkpoint['mel_scaler'];detector=m.CropDetector(a.detector_checkpoint) if a.mode=='actual' else None
    reference_file=a.detector_checkpoint or work/'reference_detector.pt'
    reference_state=torch.load(reference_file,map_location='cpu',weights_only=False);reference=float(reference_state['reference_power']);assert reference>0
    noise_meta=json.loads((raw/'runs/detector-diversity-v2/noise.json').read_text());lookup={}
    for rows in noise_meta.values():
        for r in rows:
            path=relocate(raw,r['path']);lookup[sha(path)]=path
    noises=[]
    for digest in noise_hashes:
        wave,rate=sf.read(lookup[digest],dtype='float32');assert rate==16000 and wave.ndim==1;noises.append(wave)
    originals=[]
    for record in records:
        path=source_path(raw,record,a.split);wave,rate=sf.read(path,dtype='float32')
        assert rate==16000 and wave.ndim==1
        originals.append((record,wave,LABELS.index(path.parent.name)))
    assert len(originals)==200
    @torch.inference_mode()
    def encode(wave):
        crop=detector.crop(wave) if detector else wave
        return None if crop is None else frontend.encode(crop,scaler,auxiliary=auxiliary)
    @torch.inference_mode()
    def predict(encoded):
        guesses=[]
        for start in range(0,len(encoded),64):
            items=encoded[start:start+64];valid=[i for i,x in enumerate(items) if x is not None];outputs=[10]*len(items)
            if valid:
                x,lengths,_=m.collate([items[i] for i in valid],'cuda');pred=classifier(x,lengths).argmax(1).cpu().tolist()
                for i,v in zip(valid,pred):outputs[i]=v
            guesses.extend(outputs)
        return guesses
    negative_encoded=[];negative_rows=[]
    for i,noise in enumerate(noises):
        for j in range(10):
            wave,trace=mix_levels(np.zeros(24000,np.float32),noise,[],reference,42+10000+i*100+j,[0.],vendor_mix)[0]
            negative_encoded.append(encode(wave));negative_rows.append(dict(source_id=f'negative_{i}_{j}',noise_id=f'noise{i}',noise_sha256=noise_hashes[i],waveform_sha256=hashlib.sha256(wave.tobytes()).hexdigest(),synthesis=trace))
    negative_pred=predict(negative_encoded);noise_fp=sum(x!=10 for x in negative_pred)
    write(a.output/'fixed_negative_predictions.json',[dict(r,prediction=y) for r,y in zip(negative_rows,negative_pred)])
    results=[];tic=time.monotonic()
    original_encoded=[encode(w) for _,w,_ in originals];original_pred=predict(original_encoded)
    original_metrics=m.metrics([y for _,_,y in originals],original_pred)
    results.append(dict(condition='original_already_contains_background',metrics=original_metrics,noise_fp_files=noise_fp,noise_files=80))
    write(a.output/'original_predictions.json',[dict(source_id=r['source_id'],truth=y,prediction=v) for (r,_,y),v in zip(originals,original_pred)])
    for level in a.snr_db:
        encoded=[];case_rows=[];samples=[]
        for record,wave,label in originals:
            for i,noise in enumerate(noises):
                mixed,trace=mix_levels(wave,noise,record['segments'],reference,rng_seed(f'{record["source_id"]}|{i+1}|42'),[level],vendor_mix,snr_reference=a.snr_reference)[0]
                encoded.append(encode(mixed));case_rows.append(dict(source_id=record['source_id'],noise_id=f'noise{i}',noise_sha256=noise_hashes[i],truth=label,synthesis=trace,waveform_sha256=hashlib.sha256(mixed.tobytes()).hexdigest()))
                if len(samples)<30 and len(case_rows)%53==1:samples.append(mixed)
        pred=predict(encoded);by_noise={}
        for i in range(8):
            ids=[j for j,r in enumerate(case_rows) if r['noise_id']==f'noise{i}'];assert len(ids)==200
            by_noise[f'noise{i}']=m.metrics([case_rows[j]['truth'] for j in ids],[pred[j] for j in ids])
        row=dict(condition='added_noise',nominal_snr_db=level,command10_f1=float(np.mean([v['command10_f1'] for v in by_noise.values()])),
            by_noise=by_noise,original=original_metrics,noise_fp_files=noise_fp,noise_files=80,noise_only_policy='same fixed historical train-reference0dB80 waveforms reused',snr_reference=a.snr_reference,
            forced_silence_files=sum(x is None for x in encoded),source_files=200,noisy_files=1600,selected_classifier_epoch=checkpoint['epoch'],elapsed_seconds=time.monotonic()-tic)
        other=other_gpu_jobs() if a.latency else []
        if a.latency and not other:
            times=[]
            for wave in samples[:5]:predict([encode(wave)])
            for wave in samples:
                torch.cuda.synchronize();start=time.perf_counter();predict([encode(wave)]);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
            row['latency']=dict(resident_pipeline_p50_ms=float(np.median(times)),resident_pipeline_p95_ms=float(np.percentile(times,95)),samples=len(times),device=torch.cuda.get_device_name(),
                includes='actual detector ifenabled, crop/gate, CPU mel+Burg ifaux, H2D,classifier,D2H',excludes='file IO,model loading,audio collection',lookahead_ms=80 if detector else 0,detector_frontend_lookahead_ms=12.5 if detector else 0,idle_training_process_guard=True)
        else:row['latency']=dict(measured=False,reason='other training/inference processes present' if other else '--latency not requested',other_pids=other)
        results.append(row)
        label=('snr_'+format(level,'.9f')).replace('.','p')
        write(a.output/(label+'_predictions.json'),[dict(r,prediction=y) for r,y in zip(case_rows,pred)])
        write(a.output/'metrics.json',dict(stage='running',protocol_sha256=sha(freeze),results=results))
        print('SNR_RESULT',a.split,level,'F1',row['command10_f1'],'noiseFP',noise_fp,flush=True)
    write(a.output/'metrics.json',dict(stage='complete',protocol_sha256=sha(freeze),results=results,S01_audio_read=a.split=='test',scaler_refit=False,training=False))
    print('EVALUATION_COMPLETE',a.output,flush=True)

if __name__=='__main__':main()
