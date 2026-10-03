"""Build one record per training chunk from the queue runner's own log.

Every number here was written by the runner while the run happened: the metric before
and after each chunk, the wall-clock minutes it cost, and the stop decisions. Nothing is
recomputed or estimated.
"""
import io, json, os, re

ROOT = r"D:\computer-vision-image-generation\Image generation"
LOG = os.path.join(ROOT, "logs", "_queue_runner.log")

# 2026-10-03 11:38:03    pix2pix_facades_l1 @200: val None -> 13.613 (delta n/a (first chunk))  chunk 191.4 min, 0.96 min per epoch
CHUNK = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+([a-z0-9_]+) @(\d+): val (\S+) -> (\S+)"
    r"\s+\(delta ([^)]*(?:\([^)]*\))?[^)]*)\)\s+chunk ([\d.]+) min, ([\d.]+) min per (\w+)")
STOPL = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+STOP ([a-z0-9_]+): (.+)$")
FLOOR = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+([a-z0-9_]+): gain ([-+\d.]+)% is under the ([\d.]+)% floor \((\d+)/(\d+)\)")
FAIL  = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\s+FAILED ([a-z0-9_]+) at (\d+) \(attempt (\d+)/(\d+)\)")

def num(x):
    try:
        return float(x)
    except ValueError:
        return None

chunks, stops, floors, fails = [], [], [], []
seen = set()
for ln in io.open(LOG, encoding="utf-8", errors="replace"):
    ln = ln.rstrip("\n")
    m = CHUNK.match(ln)
    if m:
        ts, mid, at, before, after, delta, mins, per, unit = m.groups()
        key = (mid, int(at), mins)
        if key in seen:          # the log double-writes some lines
            continue
        seen.add(key)
        chunks.append({"ts": ts, "model": mid, "at": int(at), "unit": unit,
                       "before": num(before), "after": num(after),
                       "minutes": float(mins), "min_per_unit": float(per)})
        continue
    m = STOPL.match(ln)
    if m: stops.append({"ts": m.group(1), "model": m.group(2), "reason": m.group(3)}); continue
    m = FLOOR.match(ln)
    if m:
        floors.append({"ts": m.group(1), "model": m.group(2), "gain_pct": float(m.group(3)),
                       "floor_pct": float(m.group(4)), "n": int(m.group(5)), "of": int(m.group(6))})
        continue
    m = FAIL.match(ln)
    if m: fails.append({"ts": m.group(1), "model": m.group(2), "at": int(m.group(3))})

state = json.load(io.open(os.path.join(ROOT, "tools", "_queue_state.json"), encoding="utf-8"))
reg = {m["id"]: m for m in json.load(
    io.open(os.path.join(ROOT, "tools", "queue_models.json"), encoding="utf-8"))["models"]}

models = {}
for mid, r in reg.items():
    st = state.get(mid, {})
    cs = sorted([c for c in chunks if c["model"] == mid], key=lambda c: c["at"])
    models[mid] = {
        "id": mid, "label": r.get("label"), "unit": r.get("unit"),
        "ceiling": r.get("total"), "chunk_size": r.get("chunk"),
        "reached": st.get("at", 0), "done": bool(st.get("done")),
        "stopped_early": bool(st.get("done")) and st.get("at", 0) < r.get("total", 0),
        "metric": st.get("best") if isinstance(st.get("best"), (int, float)) else st.get("last_val"),
        "minutes": round(sum(c["minutes"] for c in cs), 1),
        "n_chunks": len(cs),
        "chunks": cs,
        "stops": [s for s in stops if s["model"] == mid],
        "floor_hits": [f for f in floors if f["model"] == mid],
        "failures": len([f for f in fails if f["model"] == mid]),
    }

out = {"generated_from": "logs/_queue_runner.log + tools/_queue_state.json",
       "n_models": len(models), "n_chunks": len(chunks),
       "total_minutes": round(sum(m["minutes"] for m in models.values()), 1),
       "models": models}
p = r"D:\github\visual-analytics\training-dynamics\runs.json"
json.dump(out, io.open(p, "w", encoding="utf-8", newline="\n"), indent=1)
print(f"{len(models)} models, {len(chunks)} chunks, {out['total_minutes']:.0f} min total")
print(f"stops {len(stops)}  floor-hits {len(floors)}  failures {len(fails)}")
for mid, m in sorted(models.items(), key=lambda kv: -kv[1]["minutes"])[:6]:
    print(f"  {mid:20} {m['reached']:>6}/{m['ceiling']:<6} {m['minutes']:>7.0f} min  "
          f"{m['n_chunks']:>2} chunks  early={m['stopped_early']}")
