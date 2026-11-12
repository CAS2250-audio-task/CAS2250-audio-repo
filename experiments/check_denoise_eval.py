"""Independent metric recalculation, cached-wave checks, and inference replay."""
import gc
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import torch

from src import engine
from experiments.conformer.model import ConformerClassifier, load_source
from experiments.denoise_eval.filters import ROOT, STUDY, REPORT, FILTERS, make_filter
from experiments.denoise_eval.run import verify_protocol


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    engine.seed_all(42)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    protocol=verify_protocol()
    p=pd.read_csv(REPORT/"predictions.csv.gz")
    cases=pd.read_csv(REPORT/"cases.csv.gz")
    previous=pd.read_csv(ROOT/"reports/conformer-aug-v1/predictions.csv.gz")
    previous=previous[previous.condition.eq("noise") & (previous.kind.eq("clean") | previous.snr_db.eq(0))]
    pair_keys=["heldout_speaker","kind","noise_id","snr_db","draw","path","true_idx"]
    assert len(p)==83760 and len(cases)==20940
    assert not p.duplicated(["filter"]+pair_keys).any()
    assert set(p.heldout_speaker)=={"spk02","spk03","spk04","spk05"}
    assert not p.path.str.contains("spk01").any()
    none=p[p["filter"].eq("none")].merge(previous, on=pair_keys, suffixes=("","_old"),validate="one_to_one")
    assert len(none)==20940 and none.pred_idx.equals(none.pred_idx_old)
    metrics=pd.read_csv(REPORT/"condition_metrics.csv")
    group_keys=["filter","heldout_speaker","kind","noise_id","category","snr_db","draw"]
    metrics=metrics.set_index(group_keys)
    for key,g in p.groupby(group_keys):
        ref=metrics.loc[key]
        assert abs(ref.accuracy-accuracy_score(g.true_idx,g.pred_idx))<1e-12
        assert abs(ref.macro_f1-f1_score(g.true_idx,g.pred_idx,labels=list(range(11)),average="macro",zero_division=0))<1e-12
        silent=g[g.true_idx.eq(10)]
        assert abs(ref.false_command_rate-float(silent.pred_idx.ne(10).mean()))<1e-12
    summary=pd.read_csv(REPORT/"summary.csv").set_index("filter")
    for name,g in metrics.reset_index().groupby("filter"):
        assert abs(summary.loc[name,"mixed_macro_f1"]-g[g.kind.eq("mixed")].macro_f1.mean())<1e-12
        assert abs(summary.loc[name,"original_macro_f1"]-g[g.kind.eq("clean")].macro_f1.mean())<1e-12
        assert abs(summary.loc[name,"noise_only_false_command_rate"]-g[g.kind.eq("noise_only")].false_command_rate.mean())<1e-12
    state,stats=load_source(ROOT/"runs/conformer-research")
    output_hashes=0
    filter_checks=[]
    prediction_checks=0
    for name in FILTERS:
        filt=make_filter(name)
        for speaker in protocol["folds"]:
            group=p[p["filter"].eq(name)&p.heldout_speaker.eq(speaker)].sort_values("case_index")
            assert len(group)==5235
            x=np.load(STUDY/f"{speaker}_inputs.npy",mmap_mode="r")
            cache=STUDY/"filtered"/name/f"{speaker}.npy"
            y=np.load(cache,mmap_mode="r")
            assert y.shape==(5235,24000) and y.dtype==np.float32 and np.isfinite(y).all()
            audited=json.loads((cache.parent/f"{speaker}_audit.json").read_text())
            assert engine.file_hash(cache)==audited["cache_sha256"]
            hashes=[hashlib.sha256(w.tobytes()).hexdigest() for w in y]
            assert hashes==group.output_sha256.tolist()
            output_hashes+=len(y)
            # Match actual inference batch shapes: independent full16 blocks, final3.
            max_error=0.
            for offset in (0,192,1024,5232):
                replay=filt(np.array(x[offset:offset+16]))
                saved=np.array(y[offset:offset+16])
                error=float(np.max(np.abs(saved-replay)))
                assert np.allclose(saved,replay,atol=2e-5,rtol=2e-3),(name,speaker,offset,error)
                max_error=max(max_error,error)
            filter_checks.append(dict(filter=name,heldout_speaker=speaker,cases=51,max_abs_error=max_error))
            model=ConformerClassifier(state,stats,"finetune")
            checkpoint=ROOT/f"runs/conformer-aug-v1/folds/{speaker}/noise/best.pt"
            model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True)["model"],strict=True)
            model.cuda().eval()
            for offset in (0,192,1024,5216):
                batch=torch.from_numpy(np.array(y[offset:offset+32])).cuda()
                actual=model(batch).argmax(1).cpu().tolist()
                assert actual==group.pred_idx.iloc[offset:offset+len(actual)].tolist(),(name,speaker,offset)
                prediction_checks+=len(actual)
            del model,x,y
            gc.collect();torch.cuda.empty_cache()
            print("INDEPENDENT_REPLAY_OK",name,speaker,flush=True)
        del filt
        gc.collect();torch.cuda.empty_cache()
    engine.write_json(REPORT/"independent_audit.json",dict(
        rows=83760, metric_groups_recomputed=len(metrics),baseline_predictions_match=20940,
        mixture_hashes_matched_during_preparation=20128,filtered_output_sha256_checked=output_hashes,
        filter_replay_cases=sum(r["cases"] for r in filter_checks),filter_replay=filter_checks,
        classifier_replay_cases=prediction_checks,all_replayed_predictions_equal=True,
        protocol_and_source_weight_input_dataset_hashes_valid=True,test_evaluated=False,
        code_sha256=engine.file_hash(ROOT/"experiments/check_denoise_eval.py")))
    print("INDEPENDENT_DENOISE_AUDIT_OK",prediction_checks,flush=True)


if __name__=="__main__":main()
