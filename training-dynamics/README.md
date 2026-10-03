# Training dynamics

Eighteen computer-vision models trained in one queue on a single power-capped 16 GB
Quadro RTX 5000. The queue stopped each run at fixed intervals, measured a validation
metric, and recorded the cost. This is that record.

## Result

| | |
|---|---|
| models | 18 |
| GPU time | 106.7 h |
| measured intervals | 67 |
| intervals changing the metric by less than 0.2% | 41 |
| GPU time spent on those intervals | 47.8 h, **45% of the total** |

The four costliest intervals that bought nothing:

| model | interval ended at | minutes | metric change |
|---|---|---|---|
| sr_realesrgan | 20,000 steps | 242 | +0.005% |
| sr_esrgan | 20,000 steps | 213 | +0.000% |
| sr_esrgan | 17,650 steps | 209 | +0.065% |
| sr_realesrnet | 14,675 steps | 192 | +0.048% |

Two models ended measurably *worse* than a checkpoint they had already passed
(matting_u2net -0.182%, pix2pix_facades -0.268%), so their published weights come from
earlier in the run.

## Input / Output

Input: `logs/_queue_runner.log` and `tools/_queue_state.json` from the training queue.
Output: `runs.json` (per-interval records), `data.js` (chart payload), `index.html`.

## Contents

| path | what |
|---|---|
| `index.html` | the page: cost-against-gain chart, per-model table, every interval |
| `runs.json` | one record per interval, parsed from the runner log |
| `data.js` | the chart payload built from `runs.json` |
| `extract_runs.py` | the parser |
