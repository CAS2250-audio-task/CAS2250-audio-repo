"""Import the five participant ZIPs without modifying the source archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import subprocess
import unicodedata
import zipfile

import pandas as pd
import soundfile as sf

LABELS = ['출발', '정지', '전진', '후진', '왼쪽', '오른쪽', '시작', '복귀', '대기', '재시작', 'silence']


def decoded_name(info):
    name = info.filename
    if not info.flag_bits & 0x800:
        try:
            name = name.encode('cp437').decode('cp949')
        except UnicodeError:
            pass
    return unicodedata.normalize('NFC', name)


def infer_label(name):
    name = unicodedata.normalize('NFC', name)
    # Match 재시작 before 시작, and remove that match before checking ambiguity.
    matches = []
    for label in sorted(LABELS[:-1], key=len, reverse=True):
        if label in name:
            matches.append(label)
            name = name.replace(label, '')
    if len(matches) == 1:
        return matches[0]
    if not matches and ('무음' in name or 'silence' in name.lower() or Path(name).stem.endswith('Si')):
        return 'silence'
    raise ValueError(f'Unrecognized or ambiguous label: {name}')


def import_recordings(archive, output_root, seed=42, workers=4):
    import imageio_ffmpeg
    archive, out = Path(archive).resolve(), Path(output_root).resolve()
    if out.exists():
        raise FileExistsError(f'Use a fresh output directory: {out}')
    tasks = []
    with zipfile.ZipFile(archive) as outer:
        bundles = sorted((i for i in outer.infolist() if not i.is_dir()), key=lambda i: i.filename)
        if len(bundles) != 5 or any(not i.filename.endswith('.zip') for i in bundles):
            raise ValueError('Expected five participant ZIPs')
        identities = [f'spk{i+1:02}' for i in range(5)]
        shuffled = identities.copy()
        random.Random(seed).shuffle(shuffled)
        split_map = dict(zip(shuffled, ['train', 'train', 'train', 'val', 'test']))
        manifest = dict(archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                        seed=seed, split_map=split_map, participants={},
                        annotation_source='filename inference; room/session details not verified',
                        decoder='imageio-ffmpeg 0.6.0; mono 16 kHz PCM16, no gain adjustment')
        for speaker, bundle in zip(identities, bundles):
            manifest['participants'][speaker] = bundle.filename
            with zipfile.ZipFile(io.BytesIO(outer.read(bundle))) as inner:
                members = sorted((i for i in inner.infolist() if not i.is_dir()
                                  and '__MACOSX/' not in i.filename
                                  and not Path(i.filename).name.startswith('.')),
                                 key=decoded_name)
                if len(members) != 203:
                    raise ValueError(f'Expected 203 recordings: {bundle.filename}')
                for index, member in enumerate(members, 1):
                    name = decoded_name(member)
                    if Path(name).suffix.lower() != '.m4a':
                        raise ValueError(f'Unexpected audio type: {name}')
                    label = infer_label(name)
                    key = f'{speaker}_{index:03}'
                    tasks.append((dict(path=f'audio/{label}/{key}.wav', label=label,
                                       group=f'speaker:{speaker}', split=split_map[speaker],
                                       speaker_id=speaker,
                                       room_session_id=f'{speaker}:{index:03}' if label == 'silence' else '',
                                       source_filename=name, source_bundle=bundle.filename,
                                       source_sha256=hashlib.sha256(inner.read(member)).hexdigest(),
                                       notes='Label inferred from filename; session IDs are per-file identifiers'),
                                  key, inner.read(member)))
    if len(tasks) != 1015:
        raise ValueError('Expected exactly 1,015 recordings')
    out.mkdir(parents=True)
    executable = imageio_ffmpeg.get_ffmpeg_exe()

    def convert(task):
        row, key, data = task
        original = out / 'originals' / f'{key}.m4a'
        original.parent.mkdir(parents=True, exist_ok=True)
        original.write_bytes(data)
        destination = out / row['path']
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([executable, '-nostdin', '-v', 'error', '-i', str(original),
                        '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', '-threads', '1', str(destination)],
                       check=True, capture_output=True)
        info = sf.info(destination)
        if info.frames == 0:
            raise ValueError(f'Empty decoded audio: {original}')
        return dict(row, source_seconds=info.duration)

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, row in enumerate(pool.map(convert, tasks), 1):
            rows.append(row)
            if count % 100 == 0:
                print(f'Decoded {count}/{len(tasks)}', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'metadata.csv', index=False)
    (out / 'import_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(pd.crosstab(frame.label, frame.split).to_string(), flush=True)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--output-root', required=True)
    args = parser.parse_args()
    import_recordings(args.archive, args.output_root)


if __name__ == '__main__':
    main()
