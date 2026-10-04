import pytest
import torch


def test_benchmark_requires_full_stable_window_and_excludes_cold():
    from worldttn.benchmark import stable_summary
    cold = {"seconds": 100, "ranks": [{"peak_allocated_bytes": 1}]}
    samples = [{"seconds": t, "ranks": [{"peak_allocated_bytes": 10}, {"peak_allocated_bytes": 20}]} for t in (3, 5, 4)]
    report = stable_summary([cold, *samples], expected_steps=3)
    assert report["stable"]["median_seconds"] == 4
    assert report["cold"]["seconds"] == 100
    assert report["stable"]["peak_allocated_bytes"] == 20
    with pytest.raises(ValueError, match="incomplete"): stable_summary([cold, samples[0]], expected_steps=3)


def test_direct_tensor_comparison_handles_zero_reference_and_shape_errors():
    from worldttn.benchmark import tensor_difference
    r = tensor_difference(torch.ones(4), torch.zeros(4))
    assert r["relative_l2"] is None and r["max_abs"] == 1
    r = tensor_difference(torch.ones(4), torch.ones(4))
    assert r["relative_l2"] == 0 and r["within_tolerance"]
    with pytest.raises(ValueError): tensor_difference(torch.ones(4), torch.ones(3))


def test_triton_gate_uses_critical_path_and_does_not_treat_unknown_as_pass():
    from worldttn.benchmark import triton_gate
    assert triton_gate(100, .2, .1)["decision"] == "stop"
    assert triton_gate(100, 5, 2)["decision"] == "go"
    assert triton_gate(100, None, None)["decision"] == "profile_required"
    with pytest.raises(ValueError): triton_gate(100, 5, 101)


def test_strict_precision_restores_previous_flags():
    from worldttn.benchmark import precision_protocol
    before = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32
    with precision_protocol("strict"):
        assert not torch.backends.cuda.matmul.allow_tf32
    assert (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32) == before


def test_production_precision_mirrors_post_build_audit_and_restores_flags():
    from worldttn.benchmark import precision_protocol
    before = torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    audited = {"matmul_precision": "highest", "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False}
    with precision_protocol("production", audited):
        assert (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == ("highest", False, False)
    assert (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == before


def test_funnel_keeps_best_correct_candidate_below_three_percent():
    from worldttn.benchmark import select_joint_candidates
    scores = {"V0": [10, 10], "V1": [9.9, 9.9], "V2": [10.2, 10.2], "V3": [11, 11]}
    assert select_joint_candidates(scores, {"V1": True, "V2": True, "V3": True}) == ["V1"]
    assert select_joint_candidates(scores, {"V1": False, "V2": True, "V3": False}) == ["V2"]


def test_gpu_operator_entry_refuses_cpu_timing_claim(tmp_path):
    from worldttn.benchmark import operator_benchmark
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="CUDA"):
        operator_benchmark(SimpleNamespace(device="cpu", output=str(tmp_path / "out")))


def test_saved_rollout_comparison_compares_latents_states_and_psi_directly(tmp_path):
    from worldttn.benchmark import compare_saved_rollouts
    for method, delta in (("ttn_reference", 0.), ("ttn", .1)):
        torch.save({"latents": torch.ones(1, 2, 4, 1, 1) + delta,
                    "chunks": [{"chunk": 0, "start": 0, "end": 4}]}, tmp_path / f"case-000-{method}.pt")
        torch.save({"world_state": torch.ones(1, 5, 2, 8, 8) + delta,
                    "transition_fast": torch.zeros(1, 5, 2, 3) + delta,
                    "commits": 2, "predictions": 2}, tmp_path / f"case-000-{method}-state-000.pt")
    report = compare_saved_rollouts(tmp_path, 1)
    assert report[0]["generated_latents"]["max_abs"] > .09
    assert report[0]["state_chunks"][0]["anchors"][0]["psi"]["relative_l2"] is None
    assert report[0]["first_nonzero_state_difference"] == {"chunk": 0, "anchor": 3}


def test_rollout_comparison_locates_first_output_chunk_and_rejects_mismatch(tmp_path):
    from worldttn.benchmark import compare_saved_rollouts
    chunks = [{"chunk": 0, "start": 0, "end": 4}, {"chunk": 1, "start": 4, "end": 7}]
    for method in ("ttn", "ttn_reference"):
        latent = torch.ones(1, 2, 7, 1, 1)
        if method == "ttn": latent[:, :, 4:] += .1
        torch.save({"latents": latent, "chunks": chunks}, tmp_path / f"case-000-{method}.pt")
        for index in range(2):
            torch.save({"world_state": torch.ones(1, 5, 2, 8, 8), "transition_fast": torch.zeros(1, 5, 2, 3),
                        "commits": index + 2, "predictions": index + 2}, tmp_path / f"case-000-{method}-state-{index:03d}.pt")
    report = compare_saved_rollouts(tmp_path, 1)[0]
    assert report["output_chunks"][0]["latents"]["max_abs"] == 0
    assert report["output_chunks"][1]["latents"]["max_abs"] > .09
    assert report["first_nonzero_output_difference"] == {"chunk": 1, "start": 4, "end": 7}
    assert report["first_nonzero_state_difference"] is None
    torch.save({"latents": latent, "chunks": chunks[:1]}, tmp_path / "case-000-ttn.pt")
    with pytest.raises(ValueError, match="chunk"): compare_saved_rollouts(tmp_path, 1)


def test_first_cuda_call_times_the_unvalidated_invocation(monkeypatch):
    from worldttn.benchmark import first_cuda_call
    events = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    def operation(): events.append("first"); return 7
    result, seconds = first_cuda_call(operation)
    assert result == 7 and seconds >= 0 and events == ["sync", "first", "sync"]
