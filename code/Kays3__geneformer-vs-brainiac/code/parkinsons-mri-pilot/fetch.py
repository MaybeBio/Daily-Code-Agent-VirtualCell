"""Fetch the Parkinson's participants and T1 images from OpenNeuro ds005892."""
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://s3.amazonaws.com/openneuro.org/ds005892"
root = Path("data")
root.mkdir(exist_ok=True)
urlretrieve(f"{BASE}/participants.tsv", root / "participants.tsv")
with (root / "participants.tsv").open() as f:
    rows = [r for r in csv.DictReader(f, delimiter="\t") if r["group"] in {"PD-MCI", "PD-NC"}]


def fetch(row):
    participant = row["participant_id"]
    name = f"{participant}_T1w.nii.gz"
    path = root / name
    if not path.exists():
        tmp = path.with_suffix(".part")
        urlretrieve(f"{BASE}/{participant}/anat/{name}", tmp)
        tmp.replace(path)


with ThreadPoolExecutor(6) as pool:
    list(pool.map(fetch, rows))
print(f"Fetched scans for {len(rows)} participants")
