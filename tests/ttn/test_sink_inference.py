"""Public inference guards and paired input reuse for sink interventions."""
from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from worldttn.core import TTNConfig


def options(**changes):
    values = dict(tla_sink="off", sink_gain=.1, sink_position="temporal-realign", sink_start_chunk=5,
                  history_source="generated", ttn_ablation="full", eval_methods=["sana", "ttn"],
                  ttn_core_backend="reference", ttn_psi_backend="reference", camera_attention=None,
                  camera_ablation=False, ttn_compare_reference=False)
    return Namespace(**{**values, **changes})


def test_evaluation_sink_guards_and_legacy_defaults():
    from worldttn.evaluation import validate_inference_interventions
    sink = validate_inference_interventions(Namespace(), TTNConfig())
    assert sink.mode == "off"
    config = TTNConfig(stage="C", camera_attention="sana")
    assert validate_inference_interventions(options(tla_sink="protected"), config).gain == .1
    for changes, message in (
        ({"history_source": "ttn-gt"}, "TTN-only"),
        ({"tla_sink": "protected", "history_source": "gt"}, "generated"),
        ({"tla_sink": "protected", "ttn_core_backend": "reuse"}, "reference/reference"),
        ({"tla_sink": "protected", "ttn_ablation": "no-local"}, "Full Stage C"),
        ({"tla_sink": "protected", "camera_ablation": True}, "native SANA camera"),
        ({"tla_sink": "protected", "eval_methods": ["sana"]}, "requires.*TTN"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_inference_interventions(options(**changes), config)
    for config in (TTNConfig(stage="A", camera_attention="sana"), TTNConfig(stage="C")):
        with pytest.raises(ValueError, match="Full Stage C|native SANA camera"):
            validate_inference_interventions(options(tla_sink="protected"), config)
    assert validate_inference_interventions(options(history_source="ttn-gt", eval_methods=["ttn"]),
                                           TTNConfig(stage="C", camera_attention="sana")).mode == "off"


def test_cli_refuses_sink_and_mixed_history_for_training(monkeypatch, capsys):
    from worldttn import cli
    for extra in (["--tla-sink", "protected"], ["--history-source", "native-gt"]):
        monkeypatch.setattr("sys.argv", ["worldttn", "train", *extra])
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 2
        assert "evaluate-only" in capsys.readouterr().err


def test_custom_reuse_validates_protocol_and_conditioning(tmp_path):
    from tools.ttn_custom_inference import load_shared_inputs
    from worldttn.evaluation import tensor_sha256
    case = {"case_id": "custom/scene", "seed": 3407, "latent_frames": 7}
    batch = {"initial_latent": torch.zeros(1, 2, 1, 2, 2), "camera_conditions": torch.ones(1, 7, 20),
             "chunk_plucker": torch.ones(1, 3, 7, 2, 2), "y": torch.ones(1, 1, 3, 2),
             "mask": torch.ones(1, 1, 1, 3), "uncondition": torch.zeros(1, 1, 3, 2),
             "width": 64, "height": 64, "data_info": {}}
    hashes = {key: tensor_sha256(batch[key]) for key in
              ("initial_latent", "y", "mask", "uncondition", "camera_conditions", "chunk_plucker")}
    protocol = dict(case=case, prompt="scene", checkpoint_sha256="checkpoint", status="completed",
                    steps=20, cfg_scale=4.5, cached_blocks=2, cross_attn_backend="math",
                    history_source="generated", ttn_ablation="full", flow_shift=9.8,
                    execution="reference/reference", camera_attention="sana")
    episodes = [dict(method=method, input_sha256=hashes, initial_noise_sha256="noise",
                     case_id=case["case_id"], seed=3407, base_sha256="base") for method in ("sana", "ttn")]
    (tmp_path / "summary.json").write_text(json.dumps({"protocol": protocol, "episodes": episodes}))
    torch.save(batch, tmp_path / "input-bundle.pt")
    for method in ("sana", "ttn"):
        torch.save({"latents": torch.zeros(1, 2, 7, 2, 2), "method": method,
                    "case_id": case["case_id"], "seed": 3407}, tmp_path / f"case-000-{method}.pt")
    args = SimpleNamespace(steps=20, cfg_scale=4.5, cached_blocks=2, cross_attn_backend="math")
    restored, summary = load_shared_inputs(tmp_path, case, "scene", "checkpoint", args, 9.8)
    assert summary["episodes"][0]["input_sha256"] == hashes
    assert torch.equal(restored["initial_latent"], batch["initial_latent"])
    with pytest.raises(ValueError, match="protocol"):
        load_shared_inputs(tmp_path, case, "different", "checkpoint", args, 9.8)
    pair_file = tmp_path / "case-000-sana.pt"
    original = pair_file.read_bytes()
    torch.save({"latents": torch.ones(1, 2, 7, 2, 2), "method": "sana",
                "case_id": case["case_id"], "seed": 3407}, pair_file)
    with pytest.raises(ValueError, match="observed frame"):
        load_shared_inputs(tmp_path, case, "scene", "checkpoint", args, 9.8)
    pair_file.write_bytes(original)
    batch["camera_conditions"] += 1
    torch.save(batch, tmp_path / "input-bundle.pt")
    with pytest.raises(ValueError, match="conditioning"):
        load_shared_inputs(tmp_path, case, "scene", "checkpoint", args, 9.8)


@pytest.mark.parametrize("mode,gain,corrupt,expected_calls", [
    ("protected", .1, False, 1), ("protected", .1, True, 1),
    ("off", .1, False, 0), ("protected", 0., False, 0), ("zero", .1, False, 0)])
def test_rollout_verifies_reference_once_only_when_captured(monkeypatch, mode, gain, corrupt, expected_calls):
    from worldttn.cli import rollout
    from worldttn.runtime import TTNRuntimeState, TTNSystem
    from worldttn.sink import SinkOptions
    config = TTNConfig(heads=2, head_dim=8, generators=3, stage="C", camera_attention="sana")
    model = torch.nn.Module()
    model.ttn_system = TTNSystem(config)
    model.forward_long = lambda *args, **kwargs: None
    audit_calls = []
    original_verify = TTNRuntimeState.verify_sink_reference
    def verify(runtime):
        audit_calls.append(runtime.sink_reference_sha256)
        return original_verify(runtime)
    monkeypatch.setattr(TTNRuntimeState, "verify_sink_reference", verify)
    class Sampler:
        def __init__(self, *args, model_kwargs, **kwargs):
            self.session = model_kwargs["ttn_session"]
        def sample_chunks(self, noise, **kwargs):
            runtime = self.session.reset(1)
            pre = self.session.begin_chunk(0, 1, prefill=True).for_clean()
            for i in range(5): pre.stage(i, pre.predicted[:, i]+1, pre.psi[:, i], {})
            runtime.prefill(pre)
            clean = self.session.begin_chunk(0, 4).for_clean()
            for i in range(5): clean.stage(i, clean.predicted[:, i]+1, clean.psi[:, i], {})
            runtime.commit_chunk(clean)
            if corrupt: runtime.sink_reference[0, 0, 0, 0, 0] += 1
            yield 0, noise[:, :, :4], 0, 4
    monkeypatch.setitem(sys.modules, "diffusion.scheduler.self_forcing_flow_euler_sampler",
                        SimpleNamespace(SelfForcingFlowEulerCamCtrl=Sampler))
    camera = torch.cat((torch.eye(4).flatten().expand(1, 4, 16), torch.ones(1, 4, 4)), -1)
    batch = dict(initial_latent=torch.zeros(1, 1, 1, 1, 1), camera_conditions=camera,
                 width=1, height=1, y=torch.zeros(1, 1, 2, 8))
    scheduler = SimpleNamespace(scheduler=SimpleNamespace(inference_flow_shift=1.))
    def run():
        return rollout(model, scheduler, batch, steps=1, cfg_scale=1.,
                       sink_options=SinkOptions(mode, gain, "absolute", 5))
    if corrupt:
        with pytest.raises(RuntimeError, match="reference.*changed"): run()
    else:
        _, runtime, _ = run()
        assert getattr(runtime, "sink_reference_verified", None) is (True if expected_calls else None)
    assert len(audit_calls) == expected_calls
