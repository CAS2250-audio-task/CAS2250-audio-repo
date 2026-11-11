"""Prepare every recording at 1.5 s, then finalize eight human window choices."""
import argparse
import base64
import hashlib
import html
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import soundfile as sf

from .dataset import audio_path
from .review_audio import timecode

RATE = 16000
SAMPLES = 24000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_pcm(path, *, fixed=False):
    info = sf.info(path)
    if (info.samplerate, info.channels, info.subtype) != (RATE, 1, 'PCM_16'):
        raise ValueError(f'Expected mono 16-kHz PCM16: {path}')
    signal, _ = sf.read(path, dtype='int16')
    if not len(signal) or (fixed and len(signal) != SAMPLES):
        raise ValueError(f'Expected {SAMPLES} samples: {path}')
    return signal


def reduced_margin_start(length, first, last):
    """Center the complete VAD span in a 1.5-s source window."""
    if not 0 <= first < last <= length or last - first > SAMPLES or length < SAMPLES:
        raise ValueError('The full VAD span must fit inside 1.5 s')
    start = max(0, min((first + last - SAMPLES) // 2, length - SAMPLES))
    assert start <= first < last <= start + SAMPLES
    return start


def window_starts(length, regions, stride=3200, minimum_separation=800):
    """Keep endpoints and region-centered windows, adding a 200-ms grid.

    Near-duplicate starts within 50 ms are omitted. Every output is a contiguous
    24,000-sample crop; the existing gain and original internal gaps are retained.
    """
    if length <= SAMPLES or stride <= 0 or minimum_separation < 0:
        raise ValueError('Expected overlong audio and a positive stride')
    limit = length - SAMPLES
    starts = [0, limit]
    centers = [max(0, min((s['start'] + s['end'] - SAMPLES) // 2, limit)) for s in regions]
    for start in centers + list(range(0, limit + 1, stride)):
        if all(abs(start - previous) >= minimum_separation for previous in starts):
            starts.append(start)
    return sorted(set(starts))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def write_review(candidates, candidate_root, listening_root):
    """Create per-recording reels plus an offline, self-contained listening page."""
    out = Path(listening_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    sections, index, cursor, all_audio = [], [], 0, []
    gap = np.zeros(RATE // 2, dtype=np.int16)
    for group_number, (source, group) in enumerate(candidates.groupby('source_path', sort=True), 1):
        group = group.sort_values('window_number')
        first = group.iloc[0]
        title = f"{group_number:02d} · {first['label']} · {first['file_id']}"
        clips, cards = [], []
        for position, row in enumerate(group.to_dict('records')):
            path = audio_path(candidate_root, row['path'])
            signal = read_pcm(path, fixed=True)
            if index:
                all_audio.append(gap)
                cursor += len(gap)
            index.append(dict(row, reel_start_time=timecode(cursor / RATE),
                              reel_end_time=timecode((cursor + SAMPLES) / RATE),
                              reel_start_seconds=cursor / RATE))
            all_audio.append(signal)
            cursor += SAMPLES
            if clips:
                clips.append(gap)
            clips.append(signal)
            encoded = base64.b64encode(path.read_bytes()).decode('ascii')
            identity = html.escape(str(row['file_id']), quote=True)
            number = int(row['window_number'])
            cards.append(
                f'<div class="candidate"><label><input type="radio" name="{identity}" '
                f'value="{number}" onchange="update()"> 후보 {number}</label>'
                f'<audio controls preload="none" src="data:audio/wav;base64,{encoded}"></audio>'
                f'<small>원본 {row["source_start_seconds"]:.3f}–{row["source_end_seconds"]:.3f}초</small></div>')
        reel = f"{group_number:02d}_{first['label']}_{first['file_id']}_all.wav"
        sf.write(out / reel, np.concatenate(clips), RATE, subtype='PCM_16')
        reel_encoded = base64.b64encode((out / reel).read_bytes()).decode('ascii')
        sections.append(f'<section><h2>{html.escape(title)}</h2>'
                        f'<p>후보 1부터 순서대로 듣기</p><audio controls preload="none" '
                        f'src="data:audio/wav;base64,{reel_encoded}"></audio>' + ''.join(cards) + '</section>')
    sf.write(out / 'windows_all.wav', np.concatenate(all_audio), RATE, subtype='PCM_16')
    pd.DataFrame(index).to_csv(out / 'listening_index.csv', index=False, encoding='utf-8-sig')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>1.5초 윈도우 선택</title>
<style>body{font:16px system-ui;max-width:1000px;margin:32px auto;padding:0 16px;background:#f6f7fb;color:#18202d}
section{background:white;border:1px solid #dce1eb;border-radius:12px;padding:20px;margin:20px 0}
h1{font-size:26px}h2{font-size:20px}.candidate{display:flex;align-items:center;flex-wrap:wrap;gap:16px;padding:12px 0;border-top:1px solid #eee}
label{min-width:88px}small{color:#536177}audio{width:300px}textarea{width:95%;min-height:160px;font:16px monospace}button{font:inherit;padding:8px 16px}</style>
<h1>파일마다 1.5초 후보 하나 선택</h1>
<p>단어의 처음과 끝이 모두 들리는 후보를 골라 주세요. 후보들은 같은 처리본에서 추출했고 기존 증폭량을 유지합니다.</p>
<p>그룹 연속 재생에서는 후보 사이에 0.5초 간격이 있습니다. 원본 파일을 재생할 필요는 없습니다.</p>
<p>후보 번호는 파일마다 1부터 시작합니다. 선택 후 아래 내용을 채팅에 붙여 주세요. 선택 내용은 자동 전송되지 않습니다.</p>
<section><h2>선택 결과 <span id="count">0</span></h2><textarea id="result" readonly></textarea><br>
<button onclick="copyResult()">선택 결과 복사</button><p id="message"></p></section>
''' + '\n'.join(sections) + '''
<script>
function update(){const checked=[...document.querySelectorAll('input:checked')];
document.getElementById('result').value=checked.map(x=>x.name+'='+x.value).join('\n');
const total=new Set([...document.querySelectorAll('input[type=radio]')].map(x=>x.name)).size;
document.getElementById('count').textContent=checked.length+'/'+total;}
async function copyResult(){const box=document.getElementById('result');box.focus();box.select();
try{await navigator.clipboard.writeText(box.value);document.getElementById('message').textContent='복사했습니다.';}
catch(e){document.getElementById('message').textContent='선택된 내용을 Ctrl+C 또는 Cmd+C로 복사해 주세요.';}}
document.addEventListener('play',e=>{if(e.target.tagName==='AUDIO')document.querySelectorAll('audio').forEach(a=>{if(a!==e.target)a.pause();});},true);
update();
</script></html>'''
    (out / 'listen.html').write_text(page, encoding='utf-8')
    return dict(recordings=len(sections), windows=len(index), seconds=cursor / RATE,
                duration=timecode(cursor / RATE), gap_seconds=0.5)


def prepare(raw_root, accepted_root, pending_root, output_root, candidate_root, listening_root):
    raw, accepted, pending, out, review = [Path(p).resolve() for p in
                                         (raw_root, accepted_root, pending_root, output_root, candidate_root)]
    roots = [raw, accepted, pending, out, review, Path(listening_root).resolve()]
    for i, a in enumerate(roots):
        if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for b in roots[i + 1:]):
            raise ValueError('Source and output directories must be separate')
    for p in (out, review, Path(listening_root)):
        if p.exists():
            raise FileExistsError(f'Use a fresh output directory: {p}')
    metadata = pd.read_csv(raw / 'metadata.csv', dtype=str, keep_default_na=False)
    report = pd.read_csv(accepted / 'preprocess_report.csv', keep_default_na=False)
    pending_frame = pd.read_csv(pending / 'candidates.csv', keep_default_na=False)
    if metadata.path.duplicated().any() or report.path.duplicated().any() or pending_frame.path.duplicated().any():
        raise ValueError('Duplicate source identities')
    if set(metadata.path) != set(report.path) or set(pending_frame.path) != set(report[report.status == 'review'].path):
        raise ValueError('Source/report/pending identities do not match')
    report = report.set_index('path')
    pending_frame = pending_frame.set_index('path')
    out.mkdir(parents=True)
    review.mkdir(parents=True)
    records, candidates, selections = [], [], []
    for row in metadata.to_dict('records'):
        relative = row['path']
        analysis = report.loc[relative]
        source = audio_path(accepted if analysis.status == 'accepted' else pending, relative)
        signal = read_pcm(source)
        entry = dict(row, original_status=analysis.status, review_reasons=analysis.review_reasons,
                     status='prepared', method='accepted_copy' if analysis.status == 'accepted' else 'review_copy',
                     source_audio_path=os.path.relpath(source, out), source_audio_sha256=digest(source),
                     source_crop_offset_samples=0, source_crop_end_samples=len(signal),
                     prepared_sha256='')
        if len(signal) < SAMPLES:
            raise ValueError(f'Expected an already padded candidate: {source}')
        if len(signal) > SAMPLES:
            saved = pending_frame.loc[relative]
            offset = round(float(saved.candidate_crop_start_seconds) * RATE)
            regions = [dict(start=int(s['start']) - offset, end=int(s['end']) - offset)
                       for s in json.loads(analysis.speech_segments)]
            first, last = regions[0]['start'], regions[-1]['end']
            if last - first <= SAMPLES:
                start = reduced_margin_start(len(signal), first, last)
                entry.update(method='reduced_outer_margin', source_crop_offset_samples=start,
                             source_crop_end_samples=start + SAMPLES,
                             source_start_seconds=(offset + start) / RATE,
                             source_end_seconds=(offset + start + SAMPLES) / RATE,
                             left_margin_seconds=(first - start) / RATE,
                             right_margin_seconds=(start + SAMPLES - last) / RATE)
                signal = signal[start:start + SAMPLES]
            else:
                entry.update(status='awaiting_selection', method='window_selection')
                selections.append(dict(path=relative, file_id=Path(relative).stem, label=row['label'],
                                       split=row['split'], window_number='', notes=''))
                for number, start in enumerate(window_starts(len(signal), regions), 1):
                    candidate_path = Path('audio') / row['label'] / f'{Path(relative).stem}_w{number:02d}.wav'
                    destination = review / candidate_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    sf.write(destination, signal[start:start + SAMPLES], RATE, subtype='PCM_16')
                    overlaps = [max(0, min(s['end'], start + SAMPLES) - max(s['start'], start)) / RATE
                                for s in regions]
                    candidates.append(dict(source_path=relative, file_id=Path(relative).stem,
                                           label=row['label'], split=row['split'], window_number=number,
                                           path=candidate_path.as_posix(), source_start_seconds=(offset + start) / RATE,
                                           source_end_seconds=(offset + start + SAMPLES) / RATE,
                                           candidate_offset_samples=start, candidate_end_samples=start + SAMPLES,
                                           region_overlap_seconds=json.dumps(overlaps),
                                           sha256=digest(destination)))
                records.append(entry)
                continue
        destination = out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if entry['method'] == 'reduced_outer_margin':
            sf.write(destination, signal, RATE, subtype='PCM_16')
        else:
            shutil.copy2(source, destination)
        read_pcm(destination, fixed=True)
        entry['prepared_sha256'] = digest(destination)
        records.append(entry)
    metadata.to_csv(out / 'metadata_draft.csv', index=False)
    prepared = pd.DataFrame(records)
    prepared.to_csv(out / 'preparation_report.csv', index=False)
    candidates = pd.DataFrame(candidates)
    candidates.to_csv(review / 'windows.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(selections).to_csv(review / 'selections.csv', index=False, encoding='utf-8-sig')
    listening = write_review(candidates, review, listening_root)
    state = dict(status='awaiting_selection', expected_files=len(metadata), sample_rate=RATE,
                 samples=SAMPLES, seconds=1.5, candidate_root=os.path.relpath(review, out),
                 metadata_draft_sha256=digest(out / 'metadata_draft.csv'),
                 windows_manifest_sha256=digest(review / 'windows.csv'),
                 prepared_files=int((prepared.status == 'prepared').sum()),
                 awaiting_selection=int((prepared.status == 'awaiting_selection').sum()),
                 methods=prepared.method.value_counts().to_dict(), listening=listening,
                 note='All earlier review flags are retained. No metadata.csv until all window choices are supplied.')
    write_json(out / 'state.json', state)
    return state


def finalize(output_root, selections_path):
    """Require one valid human choice per recording before publishing metadata."""
    out = Path(output_root).resolve()
    state = json.loads((out / 'state.json').read_text())
    if state['status'] != 'awaiting_selection' or (out / 'metadata.csv').exists():
        raise FileExistsError('Dataset is already finalized; use a new version for revised choices')
    review = (out / state['candidate_root']).resolve()
    if digest(out / 'metadata_draft.csv') != state['metadata_draft_sha256'] or digest(review / 'windows.csv') != state['windows_manifest_sha256']:
        raise ValueError('Source metadata or window manifest changed')
    report = pd.read_csv(out / 'preparation_report.csv', keep_default_na=False)
    metadata = pd.read_csv(out / 'metadata_draft.csv', dtype=str, keep_default_na=False)
    windows = pd.read_csv(review / 'windows.csv', keep_default_na=False)
    choices = pd.read_csv(selections_path, dtype=str, keep_default_na=False)
    required = report.loc[report.status == 'awaiting_selection', 'path']
    if not {'path', 'window_number'} <= set(choices) or choices.path.duplicated().any() or set(choices.path) != set(required):
        raise ValueError('Supply exactly one selection for every pending recording')
    selected = []
    for choice in choices.to_dict('records'):
        value = choice['window_number']
        if not value.isdecimal() or int(value) < 1:
            raise ValueError(f'Missing or invalid window number: {choice["path"]}')
        match = windows[(windows.source_path == choice['path']) & (windows.window_number == int(value))]
        if len(match) != 1:
            raise ValueError(f'Unknown window: {choice["path"]}={value}')
        row = match.iloc[0].to_dict()
        source = audio_path(review, row['path'])
        read_pcm(source, fixed=True)
        if digest(source) != row['sha256']:
            raise ValueError(f'Candidate was modified: {source}')
        destination = out / choice['path']
        if destination.exists():
            raise FileExistsError(f'Existing pending destination: {destination}')
        selected.append((choice, row, source, destination))
    for row in report[report.status == 'prepared'].to_dict('records'):
        source = audio_path(out, row['path'])
        read_pcm(source, fixed=True)
        if digest(source) != row['prepared_sha256']:
            raise ValueError(f'Prepared audio was modified: {source}')
    # All choices and existing clips have passed validation before any audio is added.
    for choice, row, source, destination in selected:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        mask = report.path == choice['path']
        report.loc[mask, 'status'] = 'prepared'
        report.loc[mask, 'selected_window'] = int(choice['window_number'])
        report.loc[mask, 'selected_candidate_path'] = row['path']
        report.loc[mask, 'source_crop_offset_samples'] = row['candidate_offset_samples']
        report.loc[mask, 'source_crop_end_samples'] = row['candidate_end_samples']
        report.loc[mask, 'source_start_seconds'] = row['source_start_seconds']
        report.loc[mask, 'source_end_seconds'] = row['source_end_seconds']
        report.loc[mask, 'prepared_sha256'] = row['sha256']
    if len(metadata) != state['expected_files'] or len(list((out / 'audio').glob('*/*.wav'))) != len(metadata):
        raise ValueError('Final audio count does not match the original metadata')
    choices.to_csv(out / 'window_selections.csv', index=False, encoding='utf-8-sig')
    report.to_csv(out / 'preparation_report.csv', index=False)
    with (out / 'metadata.csv').open('x', encoding='utf-8', newline='') as f:
        metadata.to_csv(f, index=False)
    state.update(status='ready', prepared_files=len(metadata), awaiting_selection=0,
                 metadata_sha256=digest(out / 'metadata.csv'), split_counts=metadata.split.value_counts().to_dict())
    write_json(out / 'state.json', state)
    return verify_dataset(out)


def verify_dataset(output_root):
    root = Path(output_root).resolve()
    state = json.loads((root / 'state.json').read_text())
    if state['status'] != 'ready':
        raise ValueError(f"Window choices are still needed for {state['awaiting_selection']} recordings; training is not ready")
    if digest(root / 'metadata.csv') != state['metadata_sha256']:
        raise ValueError('Final metadata changed')
    metadata = pd.read_csv(root / 'metadata.csv', dtype=str, keep_default_na=False)
    report = pd.read_csv(root / 'preparation_report.csv', keep_default_na=False).set_index('path')
    if len(metadata) != state['expected_files'] or metadata.path.duplicated().any():
        raise ValueError('Final metadata count/identities changed')
    for relative in metadata.path:
        source = audio_path(root, relative)
        read_pcm(source, fixed=True)
        if digest(source) != report.loc[relative, 'prepared_sha256']:
            raise ValueError(f'Final audio was modified: {source}')
    return dict(files=len(metadata), sample_rate=RATE, samples=SAMPLES, seconds=1.5,
                split_counts=metadata.split.value_counts().to_dict(), status='ready')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--raw-root', required=True)
    p.add_argument('--accepted-root', required=True)
    p.add_argument('--pending-root', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--candidate-root', required=True)
    p.add_argument('--listening-root', required=True)
    p = sub.add_parser('finalize')
    p.add_argument('--output-root', required=True)
    p.add_argument('--selections', required=True)
    p = sub.add_parser('verify')
    p.add_argument('--output-root', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            result = prepare(args.raw_root, args.accepted_root, args.pending_root, args.output_root,
                             args.candidate_root, args.listening_root)
        elif args.command == 'finalize':
            result = finalize(args.output_root, args.selections)
        else:
            result = verify_dataset(args.output_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError) as error:
        parser.exit(2, f'Error: {error}\n')


if __name__ == '__main__':
    main()
