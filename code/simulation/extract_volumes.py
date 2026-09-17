"""Freeze the per-agent per-round feed volumes the `affect` condition delivered,
so `volume_null` can replay them exactly.

The volume-matched null is what separates "you received less" from "you received
affect-selected content". It must be an exact replay, not a matched average.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parents[2] / "data"
out = {}
for f in sorted((HERE / "runs").glob("affect_s*.json")):
    d = json.load(open(f, encoding="utf-8"))
    out[str(d["seed"])] = {str(rd["r"]): rd["volumes"] for rd in d["rounds"]}
    print(f"{f.name}: {len(out[str(d['seed'])])} rounds")
if not out:
    raise SystemExit("no affect runs yet -- run `python3 sim.py run affect` first")
json.dump(out, open(HERE / "affect_volumes.json", "w"), indent=1)
tot = sum(v for s in out.values() for r in s.values() for v in r)
n = sum(1 for s in out.values() for r in s.values() for _ in r)
print(f"wrote affect_volumes.json  mean volume {tot/n:.2f} (k=6 is the cap)")
