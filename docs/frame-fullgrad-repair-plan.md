# Frame/fullgrad repair and acceleration

The user-authorized direction is one S update per latent frame, one sequence
loss/backward, and `tbptt=0`. Native three-frame attention groups remain intact.
Keep the existing training running while preparing a separate source release.

1. Repair prefill/anchor logging and support foreground `run_config.json` in the
   metrics tool. Check the real train-result-to-logging path.
2. Expose every live checkpoint input as an explicit tensor, including selected
   memory and camera conditions. Restore device-aware L20 backward dispatch and
   pageable CPU offload. Test output, gradients, and saved-input placement.
3. Reuse protected-prefix G/B only within its live episode; invalidate on changed
   prefix/coordinates and detach only at the existing inference episode boundary.
   Reuse text K/V only inside the current block forward.
4. Suppress diagnostics during checkpoint recomputation without suppressing any
   differentiable update. Preserve first-forward frame metrics.
5. Verify history-active clips, late-loss-to-early-input gradients, compatibility,
   local CUDA where available, then independently review and package source.

No quantization, reduced solver steps, reduced history, new learned gates, or
changed objective belong to these engineering changes. Fixed-feature solve
nonexpansiveness does not establish whole-network long-rollout stability.
Performance numbers require a new benchmark; old measurements are not reused.

Progress and test evidence are recorded in the release report.
