# visual-analytics

Making models and datasets legible: training behaviour, embeddings, attribution and change over time.

| project | what | result |
|---|---|---|
| [training-dynamics](training-dynamics/) | Where 107 GPU-hours actually went across 18 training runs | 45% of the time bought <0.2% |
| [embedding-atlas](embedding-atlas/) | Three translation datasets in CLIP space | 7.2x gap between the easiest and hardest pair |

## Method

Every number is read from the artefact that produced it - a runner log, a results file,
a checkpoint - not retyped or estimated. Where a number cannot be verified it is omitted
rather than filled in.
