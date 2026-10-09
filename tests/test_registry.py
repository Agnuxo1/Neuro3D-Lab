import json
from optlab import append_run


def test_append(tmp_path):
    p = tmp_path / "runs" / "r.jsonl"
    append_run(p, {"config": {"a": 1}, "score": 0.5})
    append_run(p, {"config": {"a": 2}, "score": 0.6})
    lines = p.read_text().splitlines()
    assert len(lines) == 2
    recs = [json.loads(line) for line in lines]
    assert all(len(r["config_sha256"]) == 64 and "timestamp_utc" in r for r in recs)
    assert recs[0]["config_sha256"] != recs[1]["config_sha256"]
