"""Camera-only transition controller. Pose convention is camera-to-world SE(3)."""
import torch
from torch import nn
from .geometry import invert_se3


def skew(x):
    z = torch.zeros_like(x[..., 0])
    a, b, c = x.unbind(-1)
    return torch.stack((z, -c, b, c, z, -a, -b, a, z), -1).reshape(*x.shape[:-1], 3, 3)


def se3_log(t):
    r = t[..., :3, :3]
    cos = ((r.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    theta = cos.acos()
    vee = torch.stack((r[..., 2, 1] - r[..., 1, 2], r[..., 0, 2] - r[..., 2, 0], r[..., 1, 0] - r[..., 0, 1]), -1) / 2
    small = theta.abs() < 1e-4
    factor = torch.where(small, 1 + theta.square() / 6, theta / theta.sin().clamp_min(1e-12))
    omega = vee * factor[..., None]
    near_pi = (cos < -.9999)
    # The largest column of R+I supplies an axis even for an exact pi rotation.
    rp = r + r.transpose(-1, -2) - 2 * cos[..., None, None] * torch.eye(3, dtype=r.dtype, device=r.device)
    index = rp.square().sum(-2).argmax(-1)
    axis = rp.gather(-1, index[..., None, None].expand(*index.shape, 3, 1)).squeeze(-1)
    axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    sign = torch.where((axis * vee).sum(-1) < 0, -1., 1.)
    omega = torch.where(near_pi[..., None], axis * (theta * sign)[..., None], omega)
    o = skew(omega)
    th2 = theta.square()
    coeff = torch.where(small, 1 / 12 + th2 / 720,
                        (1 - .5 * theta / torch.tan(.5 * theta).clamp_min(1e-12)) / th2.clamp_min(1e-12))
    vinv = torch.eye(3, dtype=r.dtype, device=r.device) - .5 * o + coeff[..., None, None] * (o @ o)
    translation = (vinv @ t[..., :3, 3, None]).squeeze(-1)
    return torch.cat((translation, omega), -1)


class TransitionController(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local = nn.Sequential(nn.Linear(14, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.global_path = nn.Sequential(nn.Linear(6, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.heads = nn.ModuleList([nn.Linear(260, config.heads * config.generators) for _ in range(5)])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def features(self, poses, intrinsics, previous_pose, new_mask, width, height):
        if poses.ndim != 4 or poses.shape[-2:] != (4, 4) or intrinsics.shape != (*poses.shape[:2], 4):
            raise ValueError("poses [B,F,4,4], intrinsics [B,F,4] required")
        poses = poses.float()
        intrinsics = intrinsics.float()
        previous_pose = previous_pose.float()
        new_mask = new_mask.bool()
        count = new_mask.sum(-1).clamp_min(1)
        prior = previous_pose
        motions = []
        for f in range(poses.shape[1]):
            motions.append(se3_log(invert_se3(prior) @ poses[:, f]))
            prior = torch.where(new_mask[:, f, None, None], poses[:, f], prior)
        motion = torch.stack(motions, 1)
        tau = new_mask.cumsum(-1).float() / count[:, None]
        frequencies = 2**torch.arange(4, device=poses.device, dtype=poses.dtype) * torch.pi
        phases = tau[..., None] * frequencies
        local_in = torch.cat((motion, phases.sin(), phases.cos()), -1)
        local = (self.local(local_in) * new_mask[..., None]).sum(1) / count[:, None]
        global_features = self.global_path(se3_log(invert_se3(previous_pose) @ prior))
        scale = intrinsics.new_tensor([width, height, width, height])
        intr = (intrinsics / scale * new_mask[..., None]).sum(1) / count[:, None]
        return torch.cat((local, global_features, intr), -1)

    def forward(self, poses, intrinsics, previous_pose, new_mask, width, height):
        with torch.autocast(device_type=poses.device.type, enabled=False):
            features = self.features(poses, intrinsics, previous_pose, new_mask, width, height)
            out = torch.stack([h(features).reshape(-1, self.config.heads, self.config.generators) for h in self.heads],
                              1)
            return out * new_mask.any(-1)[:, None, None, None]
