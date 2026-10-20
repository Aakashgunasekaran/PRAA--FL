"""Shared builders for Phase 6-8 tests (no dataset access)."""


def comp(value=None, status=None, reason="evidence_missing"):
    if value is not None:
        return {"status": "valid", "value": float(value), "raw": float(value), "reason": None, "details": {}}
    if status == "degenerate":
        return {"status": "degenerate", "value": None, "raw": 1.0, "reason": reason, "details": {}}
    return {"status": "unavailable", "value": None, "raw": None, "reason": reason, "details": {}}


def row(client_id, class_id, v=None, s=None, c=None, n=10, v_status=None, s_status=None, c_status=None):
    parts = {"validation": comp(v, v_status), "stability": comp(s, s_status), "consistency": comp(c, c_status)}
    return {"client_id": client_id, "class_id": class_id, "round_id": 1, "sample_count": n,
            "prototype_dimension": 4, **parts,
            "complete": all(p["status"] == "valid" for p in parts.values())}
