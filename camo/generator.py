"""Stages 3-5: dual-branch ControlNet diffusion with mask-guided latent blending.

Per step t (Algorithm 1 of the thesis):
    eps_bg  = UNet(z_bg,  t, e_bg)                               background branch
    eps_sub = UNet(z_sub, t, e_sub, g_t * ControlNet(H_c, H_s))  subject branch
    z~_bg, z~_sub = scheduler steps
    z_out = (1 - alpha_t M) * z~_bg + alpha_t M * z~_sub
    coupling: z_bg <- z_out (bg_only) / both branches <- z_out (both)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import Config
from .imageio import format_prompt
from .preprocess import Prepared, resolve_device
from .schedules import blend_alpha, control_gate, layer_weights

ProgressFn = Callable[[int, int], None]


@dataclass
class Generation:
    image: Image.Image
    prompt: str
    seed: int
    steps_run: int
    seconds: float


@dataclass
class _Branch:
    latents: torch.Tensor
    cond: torch.Tensor
    uncond: Optional[torch.Tensor]  # None when classifier-free guidance is off
    guidance: float
    hints: Optional[List[torch.Tensor]] = None  # one tensor per ControlNet

    @property
    def rows(self) -> int:
        return self.latents.shape[0] * (1 if self.uncond is None else 2)


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    if requested == "float16":
        return torch.float16
    if requested == "float32":
        return torch.float32
    return torch.float16 if device.startswith("cuda") else torch.float32


class CamoGenerator:
    def __init__(self, sd_pipe, controlnets: Sequence, device: str, dtype: torch.dtype):
        self.sd = sd_pipe
        self.unet = sd_pipe.unet
        self.vae = sd_pipe.vae
        self.controlnets = list(controlnets)  # [canny, softedge]
        self.device = device
        self.dtype = dtype
        self.scheduler_config = sd_pipe.scheduler.config
        self.n_down = len(self.controlnets[0].controlnet_down_blocks)

    @classmethod
    def load(cls, cfg: Config) -> "CamoGenerator":
        from diffusers import ControlNetModel, StableDiffusionPipeline

        device = resolve_device(cfg.models.device)
        dtype = resolve_dtype(cfg.models.dtype, device)
        controlnets = [
            ControlNetModel.from_pretrained(repo, torch_dtype=dtype).to(device).eval()
            for repo in (cfg.models.controlnet_canny, cfg.models.controlnet_softedge)
        ]
        kwargs = dict(torch_dtype=dtype, safety_checker=None, requires_safety_checker=False)
        try:  # fp16 weights halve the download when the repo provides them
            sd = StableDiffusionPipeline.from_pretrained(
                cfg.models.base, variant="fp16" if dtype == torch.float16 else None, **kwargs
            )
        except (OSError, ValueError):
            sd = StableDiffusionPipeline.from_pretrained(cfg.models.base, **kwargs)
        sd = sd.to(device)
        sd.set_progress_bar_config(disable=True)
        if cfg.models.attention_slicing:
            sd.enable_attention_slicing()
        return cls(sd, controlnets, device, dtype)

    # ------------------------------------------------------------------
    def _make_scheduler(self, name: str, steps: int, keep_corrector: bool):
        from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, UniPCMultistepScheduler

        if name == "ddim":
            scheduler = DDIMScheduler.from_config(self.scheduler_config)
        elif name == "dpmpp_2m":
            scheduler = DPMSolverMultistepScheduler.from_config(
                self.scheduler_config, algorithm_type="dpmsolver++", solver_order=2
            )
        else:
            # UniPC's corrector rebuilds the sample from its own previous sample and
            # silently discards latents that were modified between steps (our blend).
            scheduler = UniPCMultistepScheduler.from_config(
                self.scheduler_config, disable_corrector=[] if keep_corrector else list(range(steps))
            )
        scheduler.set_timesteps(steps, device=self.device)
        return scheduler

    def _encode(self, prompt: str, negative: str, guidance: float) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        use_cfg = guidance > 1.0
        cond, uncond = self.sd.encode_prompt(prompt, self.device, 1, use_cfg, negative)
        return cond, (uncond if use_cfg else None)

    def _hint_tensor(self, hint: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.ascontiguousarray(hint)).to(self.device).float().div(255.0)
        return x.permute(2, 0, 1)[None].to(self.dtype)

    def _control_residuals(self, x, t, emb, hints: List[torch.Tensor], gate: float, cfg: Config):
        weights, mid_weight = layer_weights(cfg.control.layer_weights, self.n_down)
        scales = (cfg.control.canny_scale, cfg.control.softedge_scale)
        down_sum, mid_sum = None, None
        for net, hint, scale in zip(self.controlnets, hints, scales):
            k = float(scale) * gate
            if k == 0.0:
                continue
            down, mid = net(
                x, t, encoder_hidden_states=emb, controlnet_cond=hint.repeat(x.shape[0], 1, 1, 1), return_dict=False
            )
            down = [d * (w * k) for d, w in zip(down, weights)]
            mid = mid * (mid_weight * k)
            if down_sum is None:
                down_sum, mid_sum = down, mid
            else:
                down_sum = [a + b for a, b in zip(down_sum, down)]
                mid_sum = mid_sum + mid
        return down_sum, mid_sum

    def _predict_noise(self, branches: List[_Branch], t, gate: float, cfg: Config, scheduler) -> List[torch.Tensor]:
        """One UNet call for all branches; ControlNet residuals only reach subject rows."""
        xs, embs, controlled = [], [], []
        row = 0
        for b in branches:
            n = b.latents.shape[0]
            if b.uncond is None:
                xs.append(b.latents)
                embs.append(b.cond)
                cond_rows = all_rows = slice(row, row + n)
            else:
                xs += [b.latents, b.latents]
                embs += [b.uncond, b.cond]
                cond_rows, all_rows = slice(row + n, row + 2 * n), slice(row, row + 2 * n)
            if b.hints is not None:
                # guess mode: ControlNet only on the conditional half of CFG
                controlled.append((b, cond_rows if cfg.control.guess_mode else all_rows))
            row += b.rows

        x = scheduler.scale_model_input(torch.cat(xs), t)
        emb = torch.cat(embs)
        down_res, mid_res = None, None
        if gate > 0.0:
            for b, rows in controlled:
                down, mid = self._control_residuals(x[rows], t, emb[rows], b.hints, gate, cfg)
                if down is None:
                    continue
                if down_res is None:
                    down_res = [x.new_zeros((x.shape[0], *d.shape[1:])) for d in down]
                    mid_res = x.new_zeros((x.shape[0], *mid.shape[1:]))
                for acc, d in zip(down_res, down):
                    acc[rows] += d
                mid_res[rows] += mid

        eps = self.unet(
            x,
            t,
            encoder_hidden_states=emb,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
            return_dict=False,
        )[0]

        out, row = [], 0
        for b in branches:
            n = b.latents.shape[0]
            if b.uncond is None:
                out.append(eps[row : row + n])
            else:
                uncond, cond = eps[row : row + n], eps[row + n : row + 2 * n]
                out.append(uncond + b.guidance * (cond - uncond))
            row += b.rows
        return out

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        cfg: Config,
        prepared: Prepared,
        seed: int,
        subject_name: str,
        progress: Optional[ProgressFn] = None,
    ) -> Generation:
        started = time.time()
        s = cfg.sampling
        prompt = format_prompt(cfg.prompt.subject_template, subject_name)

        # Stage 3: prompts, schedulers, background latent initialisation
        cond_bg, uncond_bg = self._encode(cfg.prompt.background, cfg.prompt.negative, s.guidance_bg)
        cond_sub, uncond_sub = self._encode(prompt, cfg.prompt.negative, s.guidance_sub)

        bg_coupled = s.couple_mode in ("bg_only", "both")
        sub_coupled = s.couple_mode == "both"
        sched_bg = self._make_scheduler(s.scheduler, s.steps, s.legacy_unipc_corrector or not bg_coupled)
        sched_sub = self._make_scheduler(s.scheduler, s.steps, s.legacy_unipc_corrector or not sub_coupled)

        init_steps = max(1, min(s.steps, int(s.steps * s.bg_strength)))
        t_start = min(max(s.steps - init_steps, 0), s.steps - 1)
        timesteps = sched_bg.timesteps[t_start:]
        for scheduler in (sched_bg, sched_sub):
            if hasattr(scheduler, "set_begin_index"):
                scheduler.set_begin_index(t_start)

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        image = torch.from_numpy(prepared.background).to(self.device).float().div(255.0).mul(2.0).sub(1.0)
        image = image.permute(2, 0, 1)[None].to(self.dtype)
        lat_img = self.vae.encode(image).latent_dist.sample(generator=generator) * self.vae.config.scaling_factor
        noise = torch.randn(lat_img.shape, generator=generator, device=self.device, dtype=lat_img.dtype)
        lat_bg = sched_bg.add_noise(lat_img, noise, timesteps[:1])
        lat_sub = lat_bg.clone()
        if s.sub_init_noise > 0:
            extra = torch.randn(lat_sub.shape, generator=generator, device=self.device, dtype=lat_sub.dtype)
            lat_sub = lat_sub + float(s.sub_init_noise) * extra

        mask = torch.from_numpy(prepared.mask.mask)[None, None].float()
        mask = F.interpolate(mask, size=lat_bg.shape[-2:], mode="bilinear", align_corners=False)
        mask = mask.clamp(0.0, 1.0).to(self.device, self.dtype)

        bg = _Branch(lat_bg, cond_bg, uncond_bg, s.guidance_bg)
        sub = _Branch(
            lat_sub,
            cond_sub,
            uncond_sub,
            s.guidance_sub,
            hints=[self._hint_tensor(prepared.canny), self._hint_tensor(prepared.softedge)],
        )

        # Stages 4-5: dual-branch denoising + mask-guided latent blending
        total = len(timesteps)
        fused = lat_bg
        for i, t in enumerate(timesteps):
            gate = control_gate(i, total, cfg.control.start_frac, cfg.control.end_frac, cfg.control.gate)
            if cfg.models.batch_branches:
                eps_bg, eps_sub = self._predict_noise([bg, sub], t, gate, cfg, sched_bg)
            else:
                (eps_bg,) = self._predict_noise([bg], t, gate, cfg, sched_bg)
                (eps_sub,) = self._predict_noise([sub], t, gate, cfg, sched_sub)

            bg_next = sched_bg.step(eps_bg, t, bg.latents).prev_sample
            sub_next = sched_sub.step(eps_sub, t, sub.latents).prev_sample

            alpha = blend_alpha(i, total, cfg.blend.start_frac, cfg.blend.end_frac, cfg.blend.alpha_end, cfg.blend.profile)
            weight = mask * alpha
            fused = bg_next * (1.0 - weight) + sub_next * weight

            bg.latents = fused if bg_coupled else bg_next
            sub.latents = fused if sub_coupled else sub_next
            if progress is not None:
                progress(i + 1, total)

        decoded = self.vae.decode(fused / self.vae.config.scaling_factor, return_dict=False)[0]
        pixels = (decoded[0].float() / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        output = Image.fromarray((pixels * 255).round().astype(np.uint8))
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        return Generation(image=output, prompt=prompt, seed=int(seed), steps_run=total, seconds=time.time() - started)
