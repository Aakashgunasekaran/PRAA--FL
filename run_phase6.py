"""Phase 6: Prototype Reliability Score (PRS) and client contributions.

Reads a PRAM document (Phase 5 ``pram.json``; read-only) and writes ``prs.json``:

    PRS_{r,c} = (w_v V + w_s S + w_c C) / (w_v + w_s + w_c)       (documented formula)

STRICT policy (default): PRS only for records whose validation, stability and consistency
are all ``valid``; otherwise PRS is null with the reason.  Nothing is imputed.  Clients
without any PRS are ``unassessed`` and keep their FLAME weight (multiplier 1).

    python run_phase6.py --pram results/phase5/pram.json --output-dir results/phase6
    python run_phase6.py --pram results/phase5/pram.json --output-dir results/phase6_partial \
        --missing-component-policy renormalize     # EXTENSION, labelled exploratory

For the preserved Phase 3 run no client-local snapshots exist, so stability is unavailable
and strict PRS is null for every record: the honest result.  Real PRS values come from the
PRAA training run (``run_experiment.py --method praa``), whose per-round PRS/weights are in
``reliability_log.jsonl`` and which records the attested snapshots that stability needs.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from federated.pram import load_pram_output  # noqa: E402
from federated.praa import (PRAAConfig, client_reliability, compute_record_prs,  # noqa: E402
                            flame_weights, praa_client_weights, relative_multipliers)
from federated.prs import PRSConfig  # noqa: E402

SCHEMA = "praa-fl.phase6.prs/1"


def build_prs_document(pram: dict, config: PRAAConfig, pram_sha256: str = None) -> dict:
    rows = pram["records"]
    client_ids = sorted({int(r["client_id"]) for r in rows})
    prs = compute_record_prs(rows, config)
    reliability = client_reliability(prs, client_ids, config)
    multipliers, info = relative_multipliers({c: reliability[c]["reliability"] for c in client_ids}, config)
    sizes = {c: sum(int(r["sample_count"]) for r in rows if int(r["client_id"]) == c) for c in client_ids}
    base = flame_weights(sizes)
    praa = praa_client_weights(sizes, multipliers)
    scored = sum(r["prs"] is not None for r in prs)
    status = ("COMPLETE" if scored == len(prs) and prs else
              "PARTIAL_EVIDENCE" if scored else "NO_PRS_COMPUTABLE")
    return {
        "schema_version": SCHEMA,
        "phase": 6,
        "status": status,
        "policy": config.missing_component_policy,
        "extension_notice": ("renormalized PRS is an EXTENSION of the documented formula; exploratory only"
                             if config.missing_component_policy == "renormalize" else None),
        "formula": "PRS = (w_v*V + w_s*S + w_c*C)/(w_v+w_s+w_c); strict: all three components valid",
        "config": config.to_dict(),
        "source": {"pram_mode": pram.get("mode"), "pram_status": pram.get("status"),
                   "pram_formula_version": pram.get("formula_version"), "pram_sha256": pram_sha256,
                   "test_set_accessed": False},
        "summary": {"records": len(prs), "records_with_prs": scored,
                    "clients": len(client_ids),
                    "clients_assessed": sum(v["status"] == "assessed" for v in reliability.values()),
                    "multiplier_info": info},
        "clients": {str(c): {**reliability[c], "samples": sizes[c], "multiplier": multipliers[c],
                             "flame_weight": base[c], "praa_weight": praa[c]} for c in client_ids},
        "records": prs,
        "note": ("No record has a PRS: client-local snapshots (stability) do not exist for the preserved "
                 "run, so every client keeps its FLAME weight. This is a faithful result, not a bug."
                 if scored == 0 else None),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute Phase 6 PRS and client contributions")
    parser.add_argument("--pram", type=Path, default=ROOT / "results" / "phase5" / "pram.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "phase6")
    parser.add_argument("--missing-component-policy", choices=("strict", "renormalize"), default="strict")
    parser.add_argument("--client-reduce", choices=("sample_weighted", "mean"), default="sample_weighted")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--w-validation", type=float, default=1 / 3)
    parser.add_argument("--w-stability", type=float, default=1 / 3)
    parser.add_argument("--w-consistency", type=float, default=1 / 3)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    config = PRAAConfig(
        prs=PRSConfig(args.w_validation, args.w_stability, args.w_consistency),
        tau=args.tau, client_reduce=args.client_reduce,
        missing_component_policy=args.missing_component_policy)
    pram = load_pram_output(args.pram)  # validates schema
    digest = hashlib.sha256(args.pram.read_bytes()).hexdigest()
    document = build_prs_document(pram, config, digest)
    target = args.output_dir / "prs.json"
    if target.exists() and not args.overwrite:
        raise FileExistsError(f"{target} exists; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, allow_nan=False))
    s = document["summary"]
    print("=" * 60)
    print("PRAA-FL - PHASE 6: PROTOTYPE RELIABILITY SCORE (PRS)")
    print("=" * 60)
    print(f"Policy: {document['policy']}   source PRAM mode: {document['source']['pram_mode']}")
    print(f"Records: {s['records']}   with PRS: {s['records_with_prs']}")
    print(f"Clients: {s['clients']}   assessed: {s['clients_assessed']}")
    if document["note"]:
        print(document["note"])
    print("Test set accessed: False")
    print("=" * 60)
    print(f"PHASE 6 STATUS: {document['status']}   -> {target}")


if __name__ == "__main__":
    main()
