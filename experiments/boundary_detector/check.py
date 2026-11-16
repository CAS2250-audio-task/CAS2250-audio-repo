"""Synthetic-only checks; never opens team audio or starts a real experiment."""
import csv,json,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from .prepare import prepare
from .data import frame_labels
from .model import Detector,make_detector
from .segments import decode,boundary_metrics,asymmetric_cost
from .train import mix,Windows


class Checks(unittest.TestCase):
    def test_causality_shape_parameters_gradient(self):
        torch.set_num_threads(2)
        for width,count in [(32,7137),(64,22465)]:
            m=Detector(width);self.assertEqual(sum(p.numel() for p in m.parameters()),count)
            x=torch.randn(2,64,70);y=x.clone();y[:,:,40:]+=100
            self.assertTrue(torch.equal(m(x)[:,:40],m(y)[:,:40]))
            output=m(x);self.assertEqual(tuple(output.shape),(2,70));output.square().mean().backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_gru_and_asymmetric_cost(self):
        torch.set_num_threads(2)
        m=make_detector('cnn_gru');x=torch.randn(2,64,70);z=x.clone();z[:,:,40:]+=100
        torch.testing.assert_close(m(x)[:,:40],m(z)[:,:40],rtol=0,atol=0)
        m(x).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        start=asymmetric_cost([[.2,.8]],[[.3,.8]])['cost_seconds']
        end=asymmetric_cost([[.2,.8]],[[.2,.7]])['cost_seconds']
        extra=asymmetric_cost([[.2,.8]],[[.1,.8]])['cost_seconds']
        self.assertGreater(end,start);self.assertGreater(start,extra)
        self.assertAlmostEqual(asymmetric_cost([[.2,.8]],[])['cost_seconds'],3.)
        self.assertEqual(asymmetric_cost([],[[0,1]])['cost_seconds'],1.)

    def test_confirmation_backdating_and_no_margin(self):
        p=np.zeros(100);p[20:60]=1
        segments=decode(p,.01,1,on_ms=20,off_ms=50)
        np.testing.assert_allclose(segments,[[.2,.6]])
        result=boundary_metrics([[.2,.6]],segments)
        self.assertEqual(result['matched'],1);self.assertEqual(result['start_errors'],[0]);self.assertEqual(result['end_errors'],[0])
        self.assertEqual(decode(np.zeros(100),.01,1),[])

    def test_mapping_short_padding_and_split_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);sr=8000;sf.write(root/'short.wav',np.sin(np.arange(4000)*.1)*.1,sr)
            mapping=dict(path_col='recording',speaker_col='person',start_col='begin',end_col='finish',negative_col='negative',unit='samples',annotation_sample_rate=8000,annotation_source='manual')
            (root/'mapping.json').write_text(json.dumps(mapping))
            with (root/'table.csv').open('w',newline='') as f:
                w=csv.writer(f);w.writerow(['recording','person','begin','finish','negative']);w.writerow(['short.wav','dev',800,2400,'']);w.writerow(['not_uploaded_test.wav','test',0,800,''])
            records=prepare(root/'table.csv',root/'mapping.json',root/'out.jsonl',root,['dev'])
            self.assertEqual(len(records),1);self.assertEqual(records[0]['segments'],[[.1,.3]])
            np.testing.assert_array_equal(frame_labels(records[0],np.array([.09,.1,.29,.3])),[0,1,1,0])
            data=Windows([(records[0],np.ones(8000,np.float32)*.1)],[],.01)
            for index in range(4):
                x,y,valid,weights=data[index];self.assertTrue(torch.isfinite(x).all());self.assertTrue((~valid).any());self.assertEqual(len(y),151)
            # An invalid boundary is rejected rather than silently clipped.
            with (root/'table.csv').open('a') as f:f.write('short.wav,dev,3000,5000,\n')
            with self.assertRaises(ValueError):prepare(root/'table.csv',root/'mapping.json',root/'bad.jsonl',root,['dev'])

    def test_snr_uses_positive_region(self):
        rng=np.random.default_rng(7);wave=np.zeros(32000,np.float32);wave[8000:16000]=.1
        out=mix(wave,rng.normal(size=40000)*.01,[[.5,1]],rng,10,.01)
        snr=10*np.log10(np.mean(wave[8000:16000]**2)/np.mean((out-wave)[8000:16000]**2))
        self.assertAlmostEqual(snr,10,places=4)


def smoke(output=None):
    root=Path(__file__).resolve().parents[2]
    out=Path(output).resolve() if output else root/'runs/boundary-detector-synthetic-check'
    if out.exists():raise FileExistsError('Synthetic run already exists; choose a fresh test directory')
    out.mkdir(parents=True);rows=[]
    rng=np.random.default_rng(32)
    for i,(speaker,duration) in enumerate([('train',.85),('validation',1.2)]):
        wave=rng.normal(0,.001,round(duration*16000)).astype(np.float32)
        a,b=.2,duration-.15;lo,hi=round(a*16000),round(b*16000);wave[lo:hi]+=.1*np.sin(2*np.pi*(180+40*i)*np.arange(hi-lo)/16000)
        path=out/f'{speaker}.wav';sf.write(path,wave,16000)
        rows.append([path.name,speaker,a,b])
    with (out/'annotations.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['audio','speaker','start','end']);w.writerows(rows)
    mapping=dict(path_col='audio',speaker_col='speaker',start_col='start',end_col='end',unit='seconds',annotation_source='synthetic_known_interval')
    (out/'mapping.json').write_text(json.dumps(mapping))
    prepare(out/'annotations.csv',out/'mapping.json',out/'manifest.jsonl',out,['train','validation'])
    noises={}
    for kind in ['train','validation']:
        p=out/f'noise-{kind}.wav';sf.write(p,rng.normal(0,.02,40000).astype(np.float32),16000);noises[kind]=[str(p)]
    (out/'noise.json').write_text(json.dumps(noises))
    args=[sys.executable,'-m','experiments.boundary_detector.train','--manifest',str(out/'manifest.jsonl'),'--train-speakers','train','--val-speakers','validation','--noise-manifest',str(out/'noise.json'),'--output',str(out/'model'),'--epochs','2','--batch','2','--width','32','--device','cpu']
    subprocess.run(args,cwd=root,check=True)
    subprocess.run(args+['--resume'],cwd=root,check=True)
    subprocess.run([sys.executable,'-m','experiments.boundary_detector.predict','--checkpoint',str(out/'model/best.pt'),'--audio',str(out/'validation.wav'),'--output',str(out/'inference')],cwd=root,check=True)
    assert json.loads((out/'model/progress.json').read_text())['stage']=='complete'
    report=root/'reports/boundary-detector-v1';report.mkdir(parents=True,exist_ok=True)
    (report/'synthetic_checks.json').write_text(json.dumps(dict(unit_checks=5,synthetic_training_epochs=2,resume_completed_run=True,inference_export=True,real_team_audio_used=False,real_boundary_accuracy_established=False),indent=2)+'\n')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output');args=parser.parse_args()
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    smoke(args.output)
