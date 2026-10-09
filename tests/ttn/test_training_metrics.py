import gc
import json
import weakref

import pytest

from tools import ttn_training_metrics as metrics


def record(step):
    anchor = {"block": 3, "state": {"rms": .1}, "state_spectrum": {"committed": {
        "stable_rank": [[2]], "top1_energy_fraction": [[.5]]}}, "proximal": {"per_head": {
        "inner_objective_before": [[2]], "inner_objective_after": [[1]]}},
        "proximal_trajectory": [{"heldout_query": {"before": [[2]], "after_support_solve": [[1]]},
                                 "output_effect": {"relative_delta": .2}}]}
    return {"step": step, "loss": 1., "outer_grad_norm": 2., "seconds": 3., "tbptt": 4,
            "config": {"memory_update": "proximal"}, "ranks": [{"rank": 0, "commits": 2,
            "peak_allocated_bytes": 2**30, "memory_phases": [{"host_rss_bytes": 2*2**30}],
            "chunks": [{"chunk": 0, "memory_storage_bytes": 2**29, "anchors": [anchor]}]}]}


def test_source_and_formal_metrics_keep_nulls_and_expose_partial_lines(tmp_path, capsys):
    source, run = tmp_path / "source", tmp_path / "run"
    source.mkdir(); run.mkdir()
    (run / "chain.json").write_text(json.dumps({"source": str(source)}))
    (source / "train.jsonl").write_text(json.dumps(record(1)) + "\n")
    (run / "train.jsonl").write_text(json.dumps(record(2)) + "\n{partial\n")
    metrics.report(run)
    out = capsys.readouterr().out
    assert "1 | 1 | 2 | 3 | 1 | 2 | 0.5" in out
    assert "2 | 1 | 2 | 3 | 1 | 2 | 0.5" in out
    assert "1 | 3 | 0.1 | 2 | 50 | -50 | null | null | null | null | -50 | 20" in out
    assert "跳过未完整写入行" in out and "读取完成" in out


def test_rows_are_released_before_reading_next_record(tmp_path, monkeypatch):
    (tmp_path / "chain.json").write_text(json.dumps({"source": str(tmp_path)}))
    class Row(dict): pass
    def rows(root):
        first = Row(record(1))
        reference = weakref.ref(first)
        yield first
        del first
        gc.collect()
        assert reference() is None, "the report must not accumulate complete step records"
        yield Row(record(2))
    monkeypatch.setattr(metrics, "records", rows)
    metrics.report(tmp_path)


def test_empty_input_emits_progress_and_fails_instead_of_claiming_success(tmp_path, capsys):
    (tmp_path / "chain.json").write_text(json.dumps({"source": str(tmp_path)}))
    (tmp_path / "train.jsonl").touch()
    with pytest.raises(ValueError, match="没有完整训练记录"):
        metrics.report(tmp_path)
    out = capsys.readouterr().out
    assert "开始读取训练指标" in out and "读取完成" not in out


def test_mixed_offload_metrics_distinguish_copy_payload_from_rss(tmp_path, capsys):
    (tmp_path / "chain.json").write_text(json.dumps({"source": str(tmp_path)}))
    row = record(1)
    row["ranks"][0]["activation_gpu_budget_gib"] = 4
    row["ranks"][0]["memory_phases"][0]["offload_saved_tensors"] = {
        "gpu_packed_tensor_bytes": 2**29, "cpu_packed_tensor_bytes": 10*2**30}
    (tmp_path / "train.jsonl").write_text(json.dumps(row) + "\n")
    metrics.report(tmp_path)
    out = capsys.readouterr().out
    assert "累计量，不是RSS" in out and "4 | 0.5 | 10" in out
    assert "1 | 1 | 2 | 3 | 1 | 2 | 0.5" in out
