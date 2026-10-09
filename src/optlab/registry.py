# SPDX-License-Identifier: Apache-2.0
"""Append-only run log in JSON Lines format.

Each record is stamped with timestamp_utc (UTC) and config_sha256, the SHA-256 hash of
the configuration serialized as canonical JSON (sorted keys, compact separators; this
is not RFC 8785). The configuration is record['config'] if that key exists, otherwise
the whole record. The hash is computed from the input record, so it does not depend
on timestamp_utc.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os


def append_run(path_jsonl, record: dict) -> dict:
    """Append record as one JSON line to path_jsonl and return a stamped copy.

    The copy adds timestamp_utc (ISO 8601, UTC) and config_sha256. The input dict is
    not modified. Missing parent directories are created. Values that are not
    JSON-serializable are written with str().
    """
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
