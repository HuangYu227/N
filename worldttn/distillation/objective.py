"""LongLive DMD with TTN's layout, first-frame and causal-history contracts."""
from types import SimpleNamespace
import torch
from torch import nn
from longlive2.model.dmd import DMD
from worldttn.training import chunk_ranges
from .sampler import Transactions, finite, flow_x0


def to_native(value):
    return value.permute(0, 2, 1, 3, 4)


class FlowScheduler:
    """Continuous SANA flow sigma; no index lookup or second flow shift."""
    def add_noise(self, clean, noise, timestep):
        sigma = timestep.float().div(1000).reshape(-1, 1, 1, 1)
        return (1-sigma)*clean.float() + sigma*noise.float()


class Score(nn.Module):
    def __init__(self, model, batch, checkpointed=True):
        super().__init__()
        self.model, self.batch, self.checkpointed = model, batch, checkpointed

    def forward(self, noisy_image_or_video, timestep, conditional_dict, clean_x=None):
        sigma = timestep.float()/1000
        native = to_native(noisy_image_or_video)
        # Each query rebuilds its own runtime from the student's clean history.
        transaction = Transactions(self.model, self.batch, checkpointed=self.checkpointed)
        velocity = transaction.score(native, sigma, to_native(conditional_dict['history']),
                                     conditional=conditional_dict['conditional'])
        x0 = flow_x0(native, sigma[:, None, :], velocity)
        return to_native(velocity), finite(to_native(x0), 'score x0')


class StateDMD(DMD):
    """Reuse upstream CFG scoring and detached DMD target injection unchanged.

    Only TTN-specific sigma sampling and chunk normalization replace Wan's
    block reshape. Teacher/fake targets are evaluated in upstream no_grad.
    """
    def __init__(self, teacher, fake, batch, *, shift=9.8, cfg=4.5, checkpointed=True):
        nn.Module.__init__(self)
        self.device = batch['initial_latent'].device
        self.args = SimpleNamespace(i2v=True)
        self.real_score = Score(teacher, batch, checkpointed)
        self.fake_score = Score(fake, batch, checkpointed)
        self.scheduler = FlowScheduler()
        self.num_frame_per_block = 3
        self.real_guidance_scale, self.fake_guidance_scale = cfg-1, 0.
        self.timestep_shift = shift
        self.ts_schedule = self.ts_schedule_max = False
        self.min_score_timestep, self.num_train_timestep = 0, 1000
        self.min_step, self.max_step = 20, 980
        self.last_normalizers = []

    def _get_timestep(self, minimum, maximum, batch, frames, block, uniform_timestep=False):
        times = torch.empty(batch, frames, device=self.device)
        for start, end in chunk_ranges(frames):
            times[:, start:end] = torch.randint(minimum, maximum, (batch, 1), device=self.device)
        times[:, 0] = 0
        return times

    def _normalize_dmd_gradient(self, gradient, generated, real):
        finite(gradient, 'DMD direction before normalization')
        result = torch.zeros_like(gradient)
        normalizers = []
        for start, end in chunk_ranges(generated.shape[1]):
            start = max(start, 1)
            denominator = (generated[:, start:end]-real[:, start:end]).abs().mean(
                dim=(1, 2, 3, 4), keepdim=True).clamp_min(1e-6)
            result[:, start:end] = gradient[:, start:end]/denominator
            normalizers.append(denominator.flatten().detach())
        self.last_normalizers = torch.stack(normalizers, 1)
        return finite(result, 'normalized DMD direction')

    def student_loss(self, generated):
        video = to_native(generated)
        history = video.detach()
        loss, logs = self.compute_distribution_matching_loss(video,
            {'history': history, 'conditional': True}, {'history': history, 'conditional': False},
            initial_latent=video[:, :1])
        logs['normalizer_min'] = self.last_normalizers.min()
        logs['normalizer_mean'] = self.last_normalizers.mean()
        return finite(loss, 'DMD loss'), logs

    def critic_loss(self, generated):
        clean = generated.detach()
        b, _, f, _, _ = clean.shape
        times = self._get_timestep(0, 1000, b, f, 3)
        sigma = self.timestep_shift*(times/1000)/(1+(self.timestep_shift-1)*(times/1000))
        sigma = sigma.clamp(.02, .98); sigma[:, 0] = 0
        noise = torch.randn_like(clean)
        noisy = (1-sigma[:, None, :, None, None])*clean + sigma[:, None, :, None, None]*noise
        noisy[:, :, :1] = clean[:, :, :1]
        prediction, _ = self.fake_score(to_native(noisy), sigma*1000,
            {'history': to_native(clean), 'conditional': True})
        target = to_native(noise-clean)
        loss = (prediction[:, 1:].float()-target[:, 1:]).square().mean()
        return finite(loss, 'fake flow loss'), {'fake_velocity_mse': loss.detach()}
