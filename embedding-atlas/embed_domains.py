"""Embed the unpaired-translation domains with CLIP and measure how far apart they sit.

Three CycleGANs were trained on these exact pairs, so the question is answerable rather
than decorative: apple->orange, horse->zebra, summer->winter. CPU only - the GPU is
reserved for other work.

Writes embeddings, a 2D projection and per-domain statistics. The page is built
separately from what this produces.
"""
import io
import json
import os
import random
import time

import numpy as np
import torch
from PIL import Image

ROOT = os.path.join("C:", os.sep, "d drive data", "image generation dataset",
                    "gan-translation")
OUT = os.path.join("D:", os.sep, "github", "visual-analytics", "embedding-atlas")
PAIRS = {
    "apple2orange": ("apple", "orange"),
    "horse2zebra": ("horse", "zebra"),
    "summer2winter_yosemite": ("summer", "winter"),
}
PER_DOMAIN = 300
SEED = 0
THUMB = 72

random.seed(SEED)
os.makedirs(os.path.join(OUT, "thumbs"), exist_ok=True)

items = []
for pair, (na, nb) in PAIRS.items():
    for side, name in (("A", na), ("B", nb)):
        files = []
        for split in ("train", "test"):
            d = os.path.join(ROOT, pair, split + side)
            if os.path.isdir(d):
                files += [os.path.join(d, f) for f in sorted(os.listdir(d))
                          if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        random.shuffle(files)
        for p in files[:PER_DOMAIN]:
            items.append({"path": p, "pair": pair, "side": side, "domain": name})
print(f"{len(items)} images across {len(PAIRS) * 2} domains")

from transformers import CLIPModel, CLIPProcessor           # noqa: E402
MODEL = "openai/clip-vit-base-patch32"
t0 = time.perf_counter()
proc = CLIPProcessor.from_pretrained(MODEL)
model = CLIPModel.from_pretrained(MODEL).eval()
torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))   # leave the box usable
print(f"CLIP loaded in {time.perf_counter() - t0:.0f}s on CPU, "
      f"{torch.get_num_threads()} threads")

vecs, keep = [], []
BATCH = 32
t0 = time.perf_counter()
for i in range(0, len(items), BATCH):
    batch = items[i:i + BATCH]
    imgs, ok = [], []
    for it in batch:
        try:
            with Image.open(it["path"]) as im:
                im = im.convert("RGB")
                imgs.append(im.copy())
                # thumbnail for the page, written once here while the file is open
                th = im.copy()
                th.thumbnail((THUMB, THUMB), Image.LANCZOS)
                name = f"{len(keep) + len(ok):05d}.jpg"
                th.save(os.path.join(OUT, "thumbs", name), "JPEG", quality=78)
                it["thumb"] = name
                ok.append(it)
        except Exception as e:
            print(f"  skipped {os.path.basename(it['path'])}: {type(e).__name__}")
    if not ok:
        continue
    with torch.no_grad():
        inp = proc(images=imgs, return_tensors="pt")
        # transformers 5.x returns BaseModelOutputWithPooling from get_image_features,
        # not a tensor, so `.norm()` blew up 25 minutes into the run. Call the vision
        # tower and the projection directly - that signature has been stable for years.
        vis = model.vision_model(pixel_values=inp["pixel_values"])
        f = model.visual_projection(vis.pooler_output)
        f = f / f.norm(dim=-1, keepdim=True)
    vecs.append(f.numpy().astype("float32"))
    keep += ok
    if (i // BATCH) % 10 == 0:
        done = len(keep)
        rate = done / max(1e-9, time.perf_counter() - t0)
        print(f"  {done}/{len(items)}  {rate:.1f} img/s  "
              f"eta {(len(items) - done) / max(rate, 1e-9) / 60:.1f} min")

X = np.concatenate(vecs)
print(f"embedded {X.shape[0]} images in {(time.perf_counter() - t0) / 60:.1f} min")

# ---- how far apart are the two sides of each pair? --------------------------------
stats = {}
for pair in PAIRS:
    ia = [i for i, it in enumerate(keep) if it["pair"] == pair and it["side"] == "A"]
    ib = [i for i, it in enumerate(keep) if it["pair"] == pair and it["side"] == "B"]
    A, B = X[ia], X[ib]
    ca, cb = A.mean(0), B.mean(0)
    ca, cb = ca / np.linalg.norm(ca), cb / np.linalg.norm(cb)
    between = 1 - float(ca @ cb)                      # cosine distance of centroids
    within_a = 1 - float(np.mean(A @ ca))
    within_b = 1 - float(np.mean(B @ cb))
    # Separation: can a point be assigned to its own centroid rather than the other?
    corr = (np.sum((A @ ca) > (A @ cb)) + np.sum((B @ cb) > (B @ ca))) / (len(A) + len(B))
    stats[pair] = {"n_a": len(ia), "n_b": len(ib),
                   "centroid_distance": round(between, 4),
                   "spread_a": round(within_a, 4), "spread_b": round(within_b, 4),
                   "centroid_accuracy": round(float(corr), 4)}
    print(f"  {pair:24} centroid gap {between:.4f}  "
          f"spread {within_a:.3f}/{within_b:.3f}  separable {corr * 100:.1f}%")

# ---- 2D projection ----------------------------------------------------------------
from sklearn.decomposition import PCA                       # noqa: E402
from sklearn.manifold import TSNE                           # noqa: E402
t0 = time.perf_counter()
P = PCA(n_components=50, random_state=SEED).fit_transform(X)
Y = TSNE(n_components=2, perplexity=30, init="pca", random_state=SEED,
         max_iter=1000).fit_transform(P)
Y = (Y - Y.min(0)) / (Y.max(0) - Y.min(0))                  # to 0..1
print(f"projected in {time.perf_counter() - t0:.0f}s")

points = [{"x": round(float(Y[i, 0]), 4), "y": round(float(Y[i, 1]), 4),
           "p": it["pair"], "s": it["side"], "d": it["domain"], "t": it["thumb"]}
          for i, it in enumerate(keep)]
payload = {"model": MODEL, "n": len(points), "per_domain": PER_DOMAIN,
           "seed": SEED, "pairs": {k: list(v) for k, v in PAIRS.items()},
           "stats": stats, "points": points}
io.open(os.path.join(OUT, "atlas.js"), "w", encoding="utf-8", newline="\n").write(
    "window.ATLAS = " + json.dumps(payload, separators=(",", ":")) + ";\n")
json.dump({"model": MODEL, "n": len(points), "per_domain": PER_DOMAIN,
           "seed": SEED, "stats": stats},
          io.open(os.path.join(OUT, "results.json"), "w", encoding="utf-8",
                  newline="\n"), indent=2)
print("wrote atlas.js and results.json")
