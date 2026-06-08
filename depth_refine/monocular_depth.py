"""Monocular depth from RGB (Depth Anything V2) — the 'infer depth from the RGB' stage.

No retraining: uses the pretrained Depth Anything V2 weights from HuggingFace.
Returns the raw predicted (inverse) depth map, where a LARGER value = CLOSER to the
camera (disparity-like). Absolute scale is unknown (relative model), so downstream
alignment converts it to the predictor's metric scale.
"""
import numpy as np
import torch
from PIL import Image
from transformers import pipeline


class MonocularDepth:
    def __init__(self, model_name="depth-anything/Depth-Anything-V2-Small-hf", device=None):
        if device is None:
            device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("depth-estimation", model=model_name, device=device)

    @staticmethod
    def _to_pil(img01):
        """img01: (3,H,W) float tensor in [0,1] -> PIL RGB."""
        arr = (img01.detach().cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @torch.no_grad()
    def infer_batch(self, imgs01):
        """imgs01: (B,3,H,W) float [0,1]. Returns list of (H,W) float32 inverse-depth maps."""
        pil = [self._to_pil(im) for im in imgs01]
        outs = self.pipe(pil)
        if isinstance(outs, dict):  # single image safety
            outs = [outs]
        maps = []
        for o in outs:
            d = o["predicted_depth"]
            if isinstance(d, torch.Tensor):
                d = d.squeeze().detach().cpu().numpy()
            else:
                d = np.asarray(d, dtype=np.float32)
            maps.append(d.astype(np.float32))
        return maps
