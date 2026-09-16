"""CLIP-based recognizability proxies (optional; needs transformers + torch).

clip_text_gain  = cos(output, "a photo of a {animal}") - cos(background, same text)
clip_image_gain = cos(output, subject image)          - cos(background, subject image)
Positive gains mean the subject became more recognizable than in the plain background.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from PIL import Image


class ClipScorer:
    def __init__(self, model_id: str, device: str):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=self.dtype).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)

    @torch.no_grad()
    def _images(self, images) -> torch.Tensor:
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"].to(self.device, self.dtype)
        pooled = self.model.vision_model(pixel_values=pixels)[1]
        return F.normalize(self.model.visual_projection(pooled).float(), dim=-1)

    @torch.no_grad()
    def _texts(self, texts) -> torch.Tensor:
        tokens = self.processor(text=texts, return_tensors="pt", padding=True)
        pooled = self.model.text_model(
            input_ids=tokens["input_ids"].to(self.device), attention_mask=tokens["attention_mask"].to(self.device)
        )[1]
        return F.normalize(self.model.text_projection(pooled).float(), dim=-1)

    def score(self, output: Image.Image, background: Image.Image, subject: Image.Image, subject_name: str) -> Dict[str, float]:
        img = self._images([output, background, subject])
        txt = self._texts([f"a photo of a {subject_name}"])[0]
        text_sim = img @ txt
        return {
            "clip_text_output": float(text_sim[0]),
            "clip_text_gain": float(text_sim[0] - text_sim[1]),
            "clip_image_gain": float(img[0] @ img[2] - img[1] @ img[2]),
            "clip_output_vs_background": float(img[0] @ img[1]),
        }
