"""Verify (or create) the manifest of PRESERVED Phase 1-5 artefacts, and optionally verify the
large Phase 3 checkpoint that is NOT inside the ZIP.

    python scripts/verify_preserved.py                       # verify results/ against the manifest
    python scripts/verify_preserved.py --phase3-model /content/drive/MyDrive/phase3/final_model.pt
    python scripts/verify_preserved.py --write               # (maintainers) regenerate the manifest

The original Phase 3 model (final_model.pt, ~199 MB), checkpoint_round_199/200.pt and the
200-round training state live on the author's Google Drive (see docs/CHECKPOINTS.md); they are
deliberately not duplicated in the project ZIP.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "results" / "PRESERVED_MANIFEST.json"
PRESERVED = ["results/phase3/config.json", "results/phase3/metrics.json", "results/phase4/summary.json",
             "results/phase5/pram.json"]
EXTERNAL_PHASE3_MODEL_SHA256 = "cafc30c356db9f41fbf21674e040e526073d0a7645b5a5b9666fd4fe4869d2fd"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def preserved_files():
    files = list(PRESERVED) + sorted(str(p.relative_to(ROOT)) for p in (ROOT / "results" / "phase4" / "prototypes").glob("client_*.pt"))
    return files


def build_manifest() -> dict:
    return {
        "note": "SHA-256 of the preserved original artefacts. Do not regenerate unless you intend to change them.",
        "external_phase3_model": {
            "file": "final_model.pt (also checkpoint_round_200.pt, weights identical)",
            "sha256": EXTERNAL_PHASE3_MODEL_SHA256, "round": 200,
            "location": "Google Drive: MyDrive/phase3/final_model.pt  (checkpoints/checkpoint_round_199.pt, _200.pt)"},
        "files": {f: sha256(ROOT / f) for f in preserved_files()},
    }


def verify(manifest: dict) -> list:
    problems = []
    for f, expected in manifest["files"].items():
        path = ROOT / f
        if not path.exists():
            problems.append(f"missing: {f}")
        elif sha256(path) != expected:
            problems.append(f"modified: {f}")
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--phase3-model", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.write:
        MANIFEST.write_text(json.dumps(build_manifest(), indent=2))
        print(f"wrote {MANIFEST}")
        return 0
    status = 0
    if args.phase3_model:
        digest = sha256(args.phase3_model)
        ok = digest == EXTERNAL_PHASE3_MODEL_SHA256
        print(f"Phase 3 model sha256 {'MATCHES' if ok else 'DIFFERS from'} the preserved original: {digest}")
        status |= 0 if ok else 1
    problems = verify(json.loads(MANIFEST.read_text()))
    for p in problems:
        print("PROBLEM:", p)
    print(f"preserved artefacts: {'OK' if not problems else 'CHANGED'} "
          f"({len(json.loads(MANIFEST.read_text())['files'])} files checked)")
    return status | (1 if problems else 0)


if __name__ == "__main__":
    sys.exit(main())
