"""Smoke test of the dual-branch sampler with tiny random models on CPU.

Needs torch + diffusers + transformers and downloads a ~1 MB tokenizer.
Run: python tests/test_generator_tiny.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from diffusers import AutoencoderKL, ControlNetModel, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel  # noqa: E402
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer  # noqa: E402

from camo.config import Config, apply_overrides  # noqa: E402
from camo.generator import CamoGenerator  # noqa: E402
from camo.masks import build_soft_mask  # noqa: E402
from camo.preprocess import Prepared  # noqa: E402

SIZE = 64


def tiny_generator() -> CamoGenerator:
    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        block_out_channels=(32, 64), layers_per_block=2, sample_size=SIZE // 8, in_channels=4, out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"), up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=32,
    )
    vae = AutoencoderKL(
        block_out_channels=(8, 16, 16, 16), in_channels=3, out_channels=3, latent_channels=4, norm_num_groups=8,
        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
    )
    text_encoder = CLIPTextModel(CLIPTextConfig(
        bos_token_id=0, eos_token_id=2, hidden_size=32, intermediate_size=37, layer_norm_eps=1e-5,
        num_attention_heads=4, num_hidden_layers=2, pad_token_id=1, vocab_size=1000,
    ))
    tokenizer = CLIPTokenizer.from_pretrained("hf-internal-testing/tiny-random-clip")
    scheduler = PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                              skip_prk_steps=True, steps_offset=1, set_alpha_to_one=False)
    sd = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler,
                                 safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    controlnets = [ControlNetModel.from_unet(unet).eval(), ControlNetModel.from_unet(unet).eval()]
    for net in controlnets:  # from_unet zero-inits the output convs; randomise so control has an effect
        for p in list(net.controlnet_down_blocks.parameters()) + list(net.controlnet_mid_block.parameters()):
            torch.nn.init.normal_(p, std=0.02)
    return CamoGenerator(sd, controlnets, "cpu", torch.float32)


def tiny_prepared(cfg: Config) -> Prepared:
    rng = np.random.default_rng(0)
    bg = (rng.random((SIZE, SIZE, 3)) * 255).astype(np.uint8)
    soft = np.zeros((SIZE, SIZE, 3), np.uint8)
    soft[16:48, 16:48] = 220
    canny = np.zeros_like(soft)
    canny[16, 16:48] = 255
    return Prepared("s.png", "b.png", bg, bg, canny, canny, soft, build_soft_mask(soft, cfg.mask), "hed")


def make_cfg(*overrides: str) -> Config:
    cfg = Config()
    apply_overrides(cfg, [f"sampling.width={SIZE}", f"sampling.height={SIZE}", "sampling.steps=12", "sampling.bg_strength=0.6",
                          "control.canny_scale=0.5", *overrides])
    return cfg


def run(gen: CamoGenerator, *overrides: str) -> np.ndarray:
    cfg = make_cfg(*overrides)
    result = gen.generate(cfg, tiny_prepared(cfg), seed=7, subject_name="cat")
    assert result.image.size == (SIZE, SIZE) and result.steps_run == int(12 * 0.6)
    return np.asarray(result.image).astype(np.float32)


def main() -> None:
    gen = tiny_generator()
    base = run(gen)
    print("ok  default run")

    diff = np.abs(base - run(gen, "models.batch_branches=false")).max()
    assert diff <= 2.0, diff
    print(f"ok  batched branches == separate branches (max diff {diff:.2f}/255)")

    for sched in ("dpmpp_2m", "ddim"):
        run(gen, f"sampling.scheduler={sched}")
        print(f"ok  scheduler {sched}")
    for overrides in (["sampling.couple_mode=both"], ["sampling.couple_mode=none"], ["sampling.guidance_sub=1.0", "sampling.guidance_bg=1.0"],
                      ["control.guess_mode=false"], ["blend.profile=ramp"], ["sampling.sub_init_noise=0.2"]):
        run(gen, *overrides)
        print(f"ok  {' '.join(overrides)}")

    legacy_diff = np.abs(base - run(gen, "sampling.legacy_unipc_corrector=true")).mean()
    assert legacy_diff > 0.0
    print(f"ok  legacy UniPC corrector changes the blended result (mean diff {legacy_diff:.2f}/255)")

    none_fixed = run(gen, "sampling.couple_mode=none")
    none_legacy = run(gen, "sampling.couple_mode=none", "sampling.legacy_unipc_corrector=true")
    assert np.array_equal(none_fixed, none_legacy)
    print("ok  without coupling the corrector is kept (identical to legacy)")

    no_blend = run(gen, "blend.profile=constant", "blend.alpha_end=0.0")
    full_blend = run(gen, "blend.profile=constant", "blend.alpha_end=1.0")
    assert np.abs(no_blend - full_blend).mean() > 0.0
    print("ok  alpha_end changes the output")
    print("generator smoke tests passed")


if __name__ == "__main__":
    main()
