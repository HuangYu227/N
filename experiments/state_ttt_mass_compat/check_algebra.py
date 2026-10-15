"""CPU algebra checks for a design proposal, not production/model validation."""
import json
from pathlib import Path

import torch


def main():
    torch.set_num_threads(1)
    torch.manual_seed(621)
    dtype = torch.float64
    n, d = 11, 8
    eps = 1e-6
    phi = lambda x: torch.relu(x - .25)
    k, q, ko = [phi(torch.randn(n, d, dtype=dtype)) for _ in range(3)]
    v, vo = [torch.randn(n, d, dtype=dtype) for _ in range(2)]
    w = torch.rand(n, dtype=dtype) + .1
    sink = ko.T @ vo
    zo = ko.sum(0)
    state = k.T @ (w[:, None] * v)
    z = (w[:, None] * k).sum(0)
    mean = vo.mean(0, keepdim=True)
    skew = torch.randn(d, d, dtype=dtype)
    skew = skew - skew.T
    eye = torch.eye(d, dtype=dtype)
    rotation = torch.linalg.solve(eye - skew / 2, eye + skew / 2)
    coefficient = .7  # fixed within these algebra checks

    def read(query, s, reference, fallback):
        denominator = query @ (z + coefficient * zo)[:, None] + eps
        return (query @ (s + coefficient * reference) + eps * fallback) / denominator

    observed = sink.clone()
    before = read(q, state, sink, mean)
    after = read(q, state @ rotation, sink @ rotation, mean @ rotation)
    equivariance = float((after - before @ rotation).abs().max())
    assert equivariance < 1e-11
    assert torch.equal(sink, observed)

    # The correct inner objective uses the same denominator as the actual read.
    s = state.clone().requires_grad_()
    loss = lambda candidate: (
        (read(k, candidate, sink, mean) - v).square().mean()
        + .3 * (read(ko, candidate, sink, mean) - vo).square().mean()
    )
    start = loss(s)
    gradient = torch.autograd.grad(start, s, create_graph=True)[0]
    x = k / (k @ (z + coefficient * zo)[:, None] + eps)
    xo = ko / (ko @ (z + coefficient * zo)[:, None] + eps)
    curvature = 2 * (x.square().sum() + .3 * xo.square().sum()) / (n * d)
    updated = s - .5 / (curvature + eps) * gradient
    finish = loss(updated)
    assert finish < start
    meta_gradient = torch.autograd.grad(finish, s)[0]
    assert torch.isfinite(meta_gradient).all()

    # Identical values are reproduced by a consistent numerator/mass pair.
    constant = torch.randn(1, d, dtype=dtype)
    constant_output = read(q, z[:, None] * constant, zo[:, None] * constant, constant)
    constant_error = float((constant_output - constant).abs().max())
    assert constant_error < 1e-12

    # True self-removal subtracts the contribution from both numerator and mass.
    loo_error = 0.
    for i in range(n):
        self_mass = w[i] * k[i].square().sum()
        denom = k[i] @ (z + coefficient * zo) - self_mass + eps
        fast = (k[i] @ (state + coefficient * sink) - self_mass * v[i] + eps * mean[0]) / denom
        keep = torch.arange(n) != i
        reduced_state = k[keep].T @ (w[keep, None] * v[keep])
        reduced_mass = (w[keep, None] * k[keep]).sum(0)
        oracle = (k[i] @ (reduced_state + coefficient * sink) + eps * mean[0]) / (
            k[i] @ (reduced_mass + coefficient * zo) + eps
        )
        loo_error = max(loo_error, float((fast - oracle).abs().max()))
    assert loo_error < 1e-11

    # Sparse addresses do not imply sparse interference through a dense basis.
    address0 = torch.tensor([1., 0.], dtype=dtype)
    address1 = torch.tensor([0., 1.], dtype=dtype)
    basis = torch.ones(2, 1, dtype=dtype) / 2**.5
    dense_leak = float(address0 @ basis @ basis.T @ address1)
    diagonal_leak = float(address0 @ torch.eye(2, dtype=dtype) @ address1)
    assert dense_leak > .49 and diagonal_leak == 0

    # Candidate RoPE-compatible group-sparse norm lift. Components may be
    # signed, but each block kernel is nonnegative by Cauchy-Schwarz. This is
    # NOT the scalar ReLU map from the earlier six-case feature diagnostic.
    def pair_lift(raw):
        pairs = raw.reshape(*raw.shape[:-1], -1, 2)
        radius = (pairs.square().sum(-1, keepdim=True) + eps**2).sqrt()
        gate = torch.relu(radius - 1.2) / radius
        lifted = gate * torch.cat((radius, pairs), -1)
        lifted = lifted.flatten(-2)
        return lifted / lifted.square().sum(-1, keepdim=True).clamp_min(eps**2).sqrt()

    raw = torch.randn(80, d, dtype=dtype)
    angle = torch.randn(d // 2, dtype=dtype)
    c, sin = angle.cos(), angle.sin()
    rope = torch.stack((c, -sin, sin, c), -1).reshape(-1, 2, 2)
    rotated_raw = torch.einsum('bpi,pij->bpj', raw.reshape(80, -1, 2), rope).flatten(-2)
    a, ar = pair_lift(raw), pair_lift(rotated_raw)
    feature_rotation = torch.eye(3, dtype=dtype).repeat(d // 2, 1, 1)
    feature_rotation[:, 1:, 1:] = rope
    oracle_lift = torch.einsum('bpi,pij->bpj', a.reshape(80, -1, 3), feature_rotation).flatten(-2)
    rope_error = float((ar - oracle_lift).abs().max())
    kernel_min = float((a @ a.T).min())
    common_shift_error = float((a @ a.T - ar @ ar.T).abs().max())
    assert rope_error < 1e-11 and kernel_min > -1e-12 and common_shift_error < 1e-11
    lifted_rotation = torch.block_diag(*feature_rotation.unbind(0))
    bank_values = torch.randn(80, d, dtype=dtype)
    bank_content, bank_support = a.T @ bank_values, a.sum(0)
    compressed_rephase_error = float((lifted_rotation.T @ bank_content - ar.T @ bank_values).abs().max())
    support_rephase_error = float((lifted_rotation.T @ bank_support - ar.sum(0)).abs().max())
    assert compressed_rephase_error < 1e-11 and support_rephase_error < 1e-11

    # No convex-hull guarantee after an unconstrained direct-S correction.
    unconstrained = read(q, state + 1000 * torch.ones_like(state), sink, mean)
    result = dict(
        scope='FP64 small-matrix algebra only; no trained-model or rollout result',
        rotation_orthogonality_error=float((rotation.T @ rotation - eye).abs().max()),
        value_axis_transport_read_error=equivariance,
        immutable_sink_unchanged=True,
        normalized_inner_loss_before=float(start.detach()),
        normalized_inner_loss_after=float(finish.detach()),
        inner_meta_gradient_finite=True,
        constant_value_error_before_arbitrary_correction=constant_error,
        leave_one_out_error_fixed_sink_coefficient=loo_error,
        dense_projector_disjoint_address_interference=dense_leak,
        diagonal_projector_disjoint_address_interference=diagonal_leak,
        pair_lift_rope_equivariance_error=rope_error,
        pair_lift_kernel_minimum=kernel_min,
        pair_lift_common_rope_shift_kernel_error=common_shift_error,
        pair_lift_feature_dim=3 * d // 2,
        compressed_sink_rephase_content_error=compressed_rephase_error,
        compressed_sink_rephase_support_error=support_rephase_error,
        unrestricted_state_can_exceed_value_range=bool(unconstrained.abs().max() > v.abs().max()),
    )
    Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
