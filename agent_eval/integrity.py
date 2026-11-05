"""Verify the files used to reproduce the paper."""

import hashlib
import json

from .analysis import ROOT


def verify_data():
    inventory = json.loads((ROOT / "checksums.json").read_text())
    for name, expected in inventory.items():
        path = ROOT / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    print(f"Verified {len(inventory)} data, configuration and figure files.")
