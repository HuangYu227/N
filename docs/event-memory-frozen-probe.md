# Frozen-history diagnostic for event replay

This tool tests local intervention directions; it is not another training recipe
or a replacement for full-rollout quality evaluation. Existing production model,
sampler and event-memory modules are unchanged. Hooks are scoped and restored.

`tools/ttn_event_memory_probe.py capture` runs the pinned step750 forest protocol
once, maintaining an archive while returning the original state object on every
read. The complete generated latents must exactly match the historical baseline.
At sampler chunk indices 19/29/36 (event-memory IDs 20/30/37), it records the first
eligible queries and compares four top1 readers on identical baseline states:
content, initial, recent and cyclically mismatched payloads from the same archive.
Targets end at nominal 30/45/55.5 seconds, respectively.

Each CFG branch and head gets a common delta norm equal to the minimum raw norm
across the four policies, zeroed if tiny. This only scales down; it never amplifies
a weak direction. The tool records raw/matched norms, pairwise direction cosines,
candidate scores, selected/source event IDs, zero-budget coverage and bank hashes.
No ground truth enters retrieval, updates or budget selection.

Workers re-run the unchanged baseline prefix, inject the saved direction for one
policy only during the target chunk, then stop after its clean commit. They verify
the prefix hash, complete pre-target archive hash, state/query at the first actual
intervention, exact injection keys and saved per-head budgets. Future GT is used
only afterwards for that chunk's latent MSE. No later chunks are generated.

Later states/features within the target chunk naturally diverge. Consequently,
these are **fixed directions computed on a baseline**, not adaptive policy reads
from identical live queries throughout the chunk. Equal norms control one major
confound but do not equalize each head's effect after gating/projection. A latent
MSE benefit is not proof of improved RGB or long-video quality. The mismatched
control still uses valid history from the same forest.

The protocol needs one capture plus twelve truncated rollouts. Two worker
processes reuse their loaded model: GPU4 handles content/initial, GPU5 handles
recent/shuffled. Each sees one GPU, uses a 32GiB PyTorch allocation cap, and checks
at least40GiB free before loading. This is not a driver-level memory partition.
Main training on GPUs6/7 is outside this diagnostic.

Example after CPU/CUDA checks, with physical GPU restrictions set by the launcher:

```bash
CUDA_VISIBLE_DEVICES=4 python tools/ttn_event_memory_probe.py capture \
  --reference-root /path/to/original/completed/forest-protocol --output /new/probe
CUDA_VISIBLE_DEVICES=4 python tools/ttn_event_memory_probe.py worker \
  --output /new/probe --policies content initial
CUDA_VISIBLE_DEVICES=5 python tools/ttn_event_memory_probe.py worker \
  --output /new/probe --policies recent shuffled
python tools/ttn_event_memory_probe.py summarize --output /new/probe
```

The two worker commands may run in parallel after capture passes. The output must
be new; existing probe results are never overwritten. Source, checkpoint, input,
noise and frozen-direction hashes are pinned. Capture and individual probes write
status files; successful capture alone does not mean all twelve probes finished.
