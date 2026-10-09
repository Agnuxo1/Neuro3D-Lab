"""Registro append-only de ejecuciones (JSONL). Hash SHA-256 de la configuracion
(record['config'] si existe, si no el registro completo) en JSON canonico."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os


def append_run(path_jsonl, record: dict) -> dict:
    cfg = record.get("config", record)
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    out = dict(record)
    out["timestamp_utc"] = _dt.datetime.now(_dt.timezone.utc).isoformat()
    out["config_sha256"] = hashlib.sha256(canon.encode("utf-8")).hexdigest()
    d = os.path.dirname(os.fspath(path_jsonl))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path_jsonl, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(out, default=str) + "\n")
    return out
