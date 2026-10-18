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

The protocol needs one capture plus twelve truncated rollouts. In the current
deployment, both workers share physical GPU5: workerA handles content/initial,
workerB handles recent/shuffled. Each reuses its loaded model, sees one GPU,
uses a 32GiB PyTorch allocation cap and checks at least40GiB free before loading.
The launcher waits for capture to release memory and requires72GiB free before
starting both workers. This is not a driver-level memory partition. GPU4 is
reserved for another task; main training on GPUs6/7 is outside this diagnostic.

Measured on this run, two concurrent GPU5 workers each took about18.7s/chunk;
a brief single-worker interval took about5.85s/chunk. Therefore the deployment
staggers their compute, preserving both loaded models and progress. A dedicated
controller resumes workerB near workerA's last chunk, with exit/timeout/error
release paths. Two resident models do not imply a throughput advantage.

Example after CPU/CUDA checks, with physical GPU restrictions set by the launcher:

```bash
CUDA_VISIBLE_DEVICES=5 python tools/ttn_event_memory_probe.py capture \
  --reference-root /path/to/original/completed/forest-protocol --output /new/probe
CUDA_VISIBLE_DEVICES=5 python tools/ttn_event_memory_probe.py worker \
  --output /new/probe --policies content initial recent shuffled
python tools/ttn_event_memory_probe.py summarize --output /new/probe
```

The portable example uses one GPU5 worker for all policies after capture passes.
The output must be new; existing probe results are never overwritten. Source, checkpoint, input,
noise and frozen-direction hashes are pinned. Capture and individual probes write
status files; successful capture alone does not mean all twelve probes finished.
