"""Batch-one latency of the current classifier, including native Fbank features."""
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.engine import file_hash,seed_all,write_json
from experiments.conformer.model import ConformerClassifier,load_source

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"reports/conformer-latency-v1"


def measure(name,operation,warmup,repeats,cuda=False):
    for _ in range(warmup):operation()
    rows=[]
    for index in range(repeats):
        if cuda:torch.cuda.synchronize()
        start=time.perf_counter_ns()
        prediction=operation()
        if cuda:torch.cuda.synchronize()
        elapsed=(time.perf_counter_ns()-start)/1e6
        rows.append(dict(mode=name,iteration=index,milliseconds=elapsed,pred_idx=int(prediction)))
    values=np.array([row["milliseconds"] for row in rows])
    result=dict(mode=name,warmup=warmup,repeats=repeats,mean_ms=float(values.mean()),
                p50_ms=float(np.percentile(values,50)),p95_ms=float(np.percentile(values,95)),
                min_ms=float(values.min()),max_ms=float(values.max()),
                real_time_factor_p50=float(np.percentile(values,50)/1500))
    print(json.dumps(result),flush=True)
    return result,rows


@torch.inference_mode()
def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);seed_all(42)
    state,stats=load_source(ROOT/"runs/conformer-research")
    model=ConformerClassifier(state,stats,"finetune").eval()
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    checkpoint_path=ROOT/"runs/conformer-aug-v1/folds/spk02/noise/best.pt"
    checkpoint=torch.load(checkpoint_path,map_location="cpu",weights_only=True)
    model.load_state_dict(checkpoint["model"],strict=True)
    del checkpoint,state
    path=ROOT/"datasets/task2/fixed-1p5-v1/audio/출발/spk02_052.wav"
    wave,rate=sf.read(path,dtype="float32");assert rate==16000 and wave.shape==(24000,)
    host=torch.from_numpy(wave[None].copy())
    params=sum(p.numel() for p in model.parameters())
    parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
    raw=[];results=[]
    model.cuda();device_wave=host.cuda();torch.cuda.synchronize()
    expected=int(model(device_wave).argmax(1).item())
    operations=[("gpu_resident_waveform",lambda:int(model(device_wave).argmax(1).item())),
                ("gpu_host_waveform_to_label",lambda:int(model(host.to("cuda")).argmax(1).item()))]
    for name,operation in operations:
        result,rows=measure(name,operation,30,300,cuda=True)
        assert all(row["pred_idx"]==expected for row in rows)
        results.append(result);raw.extend(rows)
    gpu_allocated=torch.cuda.memory_allocated()
    gpu_peak=torch.cuda.max_memory_allocated()
    model.cpu();torch.cuda.empty_cache()
    operation=lambda:int(model(host).argmax(1).item())
    result,rows=measure("cpu_waveform_to_label_threads2",operation,10,100)
    results.append(result);raw.extend(rows)
    cpu_name=next((line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")),platform.processor())
    REPORT.mkdir(exist_ok=True)
    with (REPORT/"samples.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    audit=dict(batch_size=1,sample_rate=16000,input_samples=24000,input_seconds=1.5,
               model="Korean Conformer 12-layer classifier, no denoiser or auxiliary features",parameters=params,
               parameter_bytes=parameter_bytes,checkpoint_sha256=file_hash(checkpoint_path),
               source_sha256=file_hash(ROOT/"experiments/conformer/model.py"),
               benchmark_sha256=file_hash(Path(__file__)),input_waveform_sha256=file_hash(path),
               gpu=torch.cuda.get_device_name(0),cpu=cpu_name,cpu_threads=torch.get_num_threads(),
               cpu_interop_threads=torch.get_num_interop_threads(),torch=str(torch.__version__),
               dtype="float32",cuda_matmul_allow_tf32=True,cudnn_allow_tf32=True,
               cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
               gpu_allocated_bytes=gpu_allocated,gpu_peak_allocated_bytes=gpu_peak,
               includes="native 80-bin Fbank, fixed normalization, CNN, Conformer, pooling, classifier, argmax and CPU scalar result",
               excludes="model loading, WAV file decoding, microphone acquisition, window accumulation, networking, speech endpointing, denoising",
               measurements=results)
    write_json(REPORT/"benchmark.json",audit)
    by_name={row["mode"]:row for row in results}
    gpu=by_name["gpu_host_waveform_to_label"];cpu=by_name["cpu_waveform_to_label_threads2"]
    lines=["# 현재 Conformer 추론 지연 측정", "",f"모델 {params:,}개 파라미터. batch1, 16kHz·1.5초, float32·TF32 on.", "",
           "| 환경 | 중앙값 | p95 |", "| --- | ---: | ---: |",
           f"| {audit['gpu']}, CPU 파형→GPU→예측 라벨 | {gpu['p50_ms']:.2f}ms | {gpu['p95_ms']:.2f}ms |",
           f"| {cpu_name}, CPU 2스레드 | {cpu['p50_ms']:.2f}ms | {cpu['p95_ms']:.2f}ms |", "",
           "Native Fbank·정규화·본체·분류·argmax와 CPU로 결과를 가져오는 시간까지 포함한다. 모델 로딩·WAV 디코딩·녹음·통신·발화 끝 감지·잡음 제거는 제외한다.", "",
           "GPU는 워밍업30회 후300회, CPU는 워밍업10회 후100회 측정했다. GPU 측정 전후에 synchronize를 수행했다.", "",
           "현재는 1.5초 창 전체를 입력하는 비인과적 모델이다. 첫 창을 새로 모은다면1.5초가 별도로 필요하다. 이동창의 갱신 주기와 발화 끝 감지에 따른 지연은 별도이며, 현재 스트리밍 구현을 측정한 결과는 아니다.", "",
           "같은 파형·같은 서버의 반복 측정이다. 운영 중 전체 지연이나 다른 기기의 성능을 보장하지 않는다.", "",
           "[측정값](benchmark.json) · [전체 반복](samples.csv)", ""]
    (REPORT/"README.md").write_text("\n".join(lines))


if __name__=="__main__":main()
