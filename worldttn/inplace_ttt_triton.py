if triton is not None:
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 64, "BLOCK_C": 64, "BLOCK_H": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 128, "BLOCK_C": 64, "BLOCK_H": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 128, "BLOCK_C": 128, "BLOCK_H": 128}, num_warps=8, num_stages=3),
        ], key=["N", "H", "C"])
    @triton.jit
    def _read_add_kernel(
        z_ptr, delta_ptr, base_ptr, out_ptr, B, N, C, H,
        szb, szn, szh, sdb, sdc, sdh, sbb, sbn, sbc, sob, son, soc,
        BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        pid = tl.program_id(0)
        ntiles = tl.cdiv(N, BLOCK_N)
        ctiles = tl.cdiv(C, BLOCK_C)
        b = pid // (ntiles * ctiles)
        rem = pid % (ntiles * ctiles)
        pn = rem // ctiles
        pc = rem % ctiles
        n = pn * BLOCK_N + tl.arange(0, BLOCK_N)
        c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
        acc = tl.zeros((BLOCK_N, BLOCK_C), tl.float32)
        for hs in tl.range(0, tl.cdiv(H, BLOCK_H)):
            h = hs * BLOCK_H + tl.arange(0, BLOCK_H)
            z = tl.load(z_ptr + b * szb + n[:, None] * szn + h[None, :] * szh,
                        mask=(n < N)[:, None] & (h < H)[None, :], other=0.0)
            d = tl.load(delta_ptr + b * sdb + c[:, None] * sdc + h[None, :] * sdh,
                        mask=(c < C)[:, None] & (h < H)[None, :], other=0.0)
            acc += tl.dot(z, tl.trans(d))
        base = tl.load(base_ptr + b * sbb + n[:, None] * sbn + c[None, :] * sbc,
                       mask=(n < N)[:, None] & (c < C)[None, :], other=0.0).to(tl.float32)
        tl.store(out_ptr + b * sob + n[:, None] * son + c[None, :] * soc,
                 base + acc, mask=(n < N)[:, None] & (c < C)[None, :])

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_N": 64, "BLOCK_C": 64, "BLOCK_H": 64}, num_warps=4, num_stages=2),
            triton.Config({"BLOCK_N": 128, "BLOCK_C": 64, "BLOCK_H": 128}, num_warps=8, num_stages=2),
            triton.Config({"BLOCK_N": 128, "BLOCK_C": 128, "BLOCK_H": 128}, num_warps=8, num_stages=3),
        ], key=["N", "H", "C"])
    @triton.jit
    def _update_kernel(
        v_ptr, z_ptr, delta_ptr, out_ptr, scale_ptr, B, N, C, H,
        svb, svn, svc, szb, szn, szh, sdb, sdc, sdh, sob, soc, soh,
        BLOCK_N: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr,
    ):
        pid = tl.program_id(0)
        ctiles = tl.cdiv(C, BLOCK_C)
        htiles = tl.cdiv(H, BLOCK_H)
        b = pid // (ctiles * htiles)
        rem = pid % (ctiles * htiles)
        pc = rem // htiles
        ph = rem % htiles
        c = pc * BLOCK_C + tl.arange(0, BLOCK_C)
        h = ph * BLOCK_H + tl.arange(0, BLOCK_H)
        acc = tl.zeros((BLOCK_C, BLOCK_H), tl.float32)
        for n0 in tl.range(0, N, BLOCK_N):
            n = n0 + tl.arange(0, BLOCK_N)
            v = tl.load(v_ptr + b * svb + n[:, None] * svn + c[None, :] * svc,
                        mask=(n < N)[:, None] & (c < C)[None, :], other=0.0)
            z = tl.load(z_ptr + b * szb + n[:, None] * szn + h[None, :] * szh,
                        mask=(n < N)[:, None] & (h < H)[None, :], other=0.0)
            acc += tl.dot(tl.trans(v), z)
        m = (c < C)[:, None] & (h < H)[None, :]
        old = tl.load(delta_ptr + b * sdb + c[:, None] * sdc + h[None, :] * sdh, mask=m, other=0.0)
        scale = tl.load(scale_ptr + b)
        tl.store(out_ptr + b * sob + c[:, None] * soc + h[None, :] * soh, old + scale * acc, mask=m)

def read_add(z, delta, base):
    if not available():
        return (base.float() + torch.bmm(z.float(), delta.transpose(1, 2))).to(base.dtype)
    B, N, H = z.shape
    C = delta.shape[1]
    out = torch.empty_like(base)
    grid = (B * triton.cdiv(N, 128) * triton.cdiv(C, 128),)
    _read_add_kernel[grid](
        z, delta, base, out, B, N, C, H,
        z.stride(0), z.stride(1), z.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        base.stride(0), base.stride(1), base.stride(2),
        out.stride(0), out.stride(1), out.stride(2))
    return out

def update(v, z, delta, scale):
    if not available():
        return delta + scale[:, None, None] * torch.bmm(v.transpose(1, 2), z)
    B, N, C = v.shape
    H = z.shape[-1]
    out = torch.empty_like(delta)
    grid = (B * triton.cdiv(C, 128) * triton.cdiv(H, 128),)
    _update_kernel[grid](
        v, z, delta, out, scale, B, N, C, H,
        v.stride(0), v.stride(1), v.stride(2),
        z.stride(0), z.stride(1), z.stride(2),
        delta.stride(0), delta.stride(1), delta.stride(2),
        out.stride(0), out.stride(1), out.stride(2))
    return out
