# Embedding atlas

Three CycleGANs were trained to translate between domains: apple ↔ orange,
horse ↔ zebra, summer ↔ winter. This embeds 300 images from each of the six domains
with CLIP ViT-B/32 and measures how far apart the two sides of each pair actually sit.

## Result

| dataset | centroid distance | assigned to own domain |
|---|---|---|
| horse → zebra | 0.1375 | 99.0% |
| apple → orange | 0.0865 | 98.7% |
| summer → winter | 0.0191 | 83.7% |

Summer/winter's domains sit 7.2× closer together than horse/zebra's, and one image in
six is closer to the other domain's centroid than to its own.

Distances are computed on the raw 512-dimensional vectors, not on the 2D projection —
a projection distorts distance, so measuring on it would measure the projection.

**Three pairs cannot establish a relationship.** The domain gap is what is measured here.
Whether a smaller gap makes translation harder needs many more than three pairs.

## Input / Output

Input: 1,800 images, 300 per domain, from the three unpaired-translation datasets.
Output: `results.json` (per-pair statistics), `atlas.js` (2D projection + thumbnails),
`index.html` (interactive atlas).

## Contents

| path | what |
|---|---|
| `index.html` | the atlas: hover any point to see the image |
| `results.json` | centroid distance, spread and separability per pair |
| `embed_domains.py` | CLIP embedding, statistics and t-SNE projection |
