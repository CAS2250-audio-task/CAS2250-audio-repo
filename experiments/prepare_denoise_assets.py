"""Restore pinned official inference code/weights without changing existing assets."""
import hashlib
import io
import json
from pathlib import Path
import ssl
import tarfile
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "runs/denoise-eval-v1"
REPORT = ROOT / "reports/denoise-eval-v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def download(url, expected=None):
    try:
        response = urllib.request.urlopen(url, timeout=90)
    except urllib.error.URLError as error:
        # This one public host had an expired certificate during the original run.
        # Permit fallback only for this exact public weight URL and a full pinned hash.
        if (url != "https://dl.fbaipublicfiles.com/adiyoss/denoiser/dns64-a7761ff99a7d5bb6.th"
                or expected is None or "certificate has expired" not in str(error)):
            raise
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        response = urllib.request.urlopen(url, context=context, timeout=90)
    with response:
        data = response.read()
    if expected is not None and sha(data) != expected:
        raise ValueError(f"Downloaded hash mismatch: {url}")
    return data


def main():
    protocol = json.loads((REPORT / "protocol.json").read_text())
    pins = protocol["vendor_sha256"]
    missing = set()
    for path, expected in pins.items():
        target = ROOT / path
        if target.exists():
            if sha(target.read_bytes()) != expected:
                raise ValueError(f"Existing pinned asset changed: {path}")
        else:
            missing.add(path)

    def put(path, data):
        if path not in missing:
            return
        assert sha(data) == pins[path], path
        target = ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        missing.remove(path)

    for name in ("DeepFilterNet", "denoiser"):
        prefix = "runs/denoise-eval-v1/vendor/"+name+"/"
        if not any(p.startswith(prefix) for p in missing):
            continue
        meta = protocol["official_revisions"][name]
        data = download(f"https://api.github.com/repos/{meta['repo']}/tarball/{meta['revision']}")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile() and "/" in member.name:
                    relative = member.name.split("/", 1)[1]
                    put(prefix+relative, archive.extractfile(member).read())
    revision = protocol["official_revisions"]["nsnet2"]["revision"]
    prefix = "runs/denoise-eval-v1/vendor/nsnet2/"
    for path in list(missing):
        if path.startswith(prefix):
            name = path[len(prefix):]
            url = f"https://raw.githubusercontent.com/microsoft/DNS-Challenge/{revision}/NSNet2-baseline/{name}"
            put(path, download(url, pins[path]))
    if any(p.startswith("runs/denoise-eval-v1/models/DeepFilterNet3/") for p in missing):
        rev = protocol["official_revisions"]["DeepFilterNet"]["revision"]
        url = f"https://raw.githubusercontent.com/Rikorose/DeepFilterNet/{rev}/models/DeepFilterNet3.zip"
        data = download(url, "49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in archive.namelist():
                if not name.endswith("/"):
                    put("runs/denoise-eval-v1/models/"+name, archive.read(name))
    for path in list(missing):
        if "/models/dns64/" in path:
            url = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/"+Path(path).name
            put(path, download(url, pins[path]))
    if missing:
        raise ValueError(f"Unrecognized pinned assets: {sorted(missing)}")
    preflight = REPORT / "preflight.json"
    assert sha(preflight.read_bytes()) == protocol["preflight_sha256"]
    destination = STUDY / "preflight.json"
    if not destination.exists():
        destination.write_bytes(preflight.read_bytes())
    print("PINNED_DENOISER_ASSETS_VALID", len(pins))


if __name__ == "__main__":
    main()
