"""Text-prompted instance segmenters behind one interface.

- `Sam3Segmenter`: SAM 3 (transformers `Sam3Model`), promptable concept segmentation from text.
- `GroundedSam2Segmenter`: Grounding DINO boxes from text -> SAM 2.1 masks from those boxes.
- `make_segmenter("auto")`: SAM 3 if its weights load, else the Grounding DINO + SAM 2.1 fallback.

Models load lazily on first use, in bf16 on the GPU when available, and `make_segmenter` keeps one
instance per name per process (the GPU is shared with other workers).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

log = logging.getLogger(__name__)

SAM3_REPO = os.environ.get("MOTORMEM_SAM3", "facebook/sam3")
SAM2_REPO = os.environ.get("MOTORMEM_SAM2", "facebook/sam2.1-hiera-large")
GDINO_REPOS = ("IDEA-Research/grounding-dino-base", "IDEA-Research/grounding-dino-tiny")


@dataclass
class MaskCandidate:
    mask: np.ndarray        # (H, W) bool, image convention
    label: str              # the prompt that produced it
    score: float            # detector confidence in [0, 1]
    box: np.ndarray         # (4,) float xyxy pixels


@runtime_checkable
class Segmenter(Protocol):
    name: str

    def segment(self, rgb: np.ndarray, prompts: list[str]) -> list[MaskCandidate]: ...


# ---------------------------------------------------------------- helpers
def _torch():
    import torch
    return torch


def _device_dtype():
    torch = _torch()
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def _is_cached(repo: str) -> bool:
    """True if `repo` has a transformers config in the local HF cache (no network)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(repo, "config.json")
        return isinstance(p, str) and os.path.exists(p)
    except Exception:
        return False


def mask_box(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros(4)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N,4) and (M,4) xyxy boxes."""
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(-1)
    area = lambda x: np.clip(x[:, 2:] - x[:, :2], 0, None).prod(-1)
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.5) -> list[int]:
    order = list(np.argsort(-scores))
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(int(i))
        if order:
            ious = _box_iou(boxes[[i]], boxes[order])[0]
            order = [j for j, o in zip(order, ious) if o <= iou]
    return keep


# ---------------------------------------------------------------- SAM 3
class Sam3Segmenter:
    """SAM 3 text-prompted instance segmentation. Vision features are computed once per image."""

    name = "sam3"

    def __init__(self, repo: str = SAM3_REPO, threshold: float = 0.4, mask_threshold: float = 0.5,
                 local_only: bool = True):
        self.repo, self.threshold, self.mask_threshold = repo, threshold, mask_threshold
        self.local_only = local_only
        self._model = self._proc = None
        self._lock = threading.Lock()

    def load(self) -> "Sam3Segmenter":
        with self._lock:
            if self._model is None:
                from transformers import Sam3Model, Sam3Processor
                dev, dt = _device_dtype()
                t0 = time.time()
                self._proc = Sam3Processor.from_pretrained(self.repo, local_files_only=self.local_only)
                self._model = Sam3Model.from_pretrained(
                    self.repo, dtype=dt, local_files_only=self.local_only).to(dev).eval()
                log.info("SAM 3 loaded from %s on %s in %.1fs", self.repo, dev, time.time() - t0)
        return self

    def segment(self, rgb: np.ndarray, prompts: list[str]) -> list[MaskCandidate]:
        self.load()
        torch = _torch()
        from PIL import Image
        dev = self._model.device
        dt = self._model.dtype
        img = Image.fromarray(np.ascontiguousarray(rgb))
        out: list[MaskCandidate] = []
        with torch.inference_mode():
            inputs = self._proc(images=img, return_tensors="pt").to(dev)
            vision = self._model.get_vision_features(pixel_values=inputs["pixel_values"].to(dt))
            sizes = inputs["original_sizes"].tolist()
            for p in prompts:
                t = self._proc(text=p, return_tensors="pt").to(dev)
                o = self._model(vision_embeds=vision, input_ids=t["input_ids"],
                                attention_mask=t["attention_mask"])
                r = self._proc.post_process_instance_segmentation(
                    o, threshold=self.threshold, mask_threshold=self.mask_threshold, target_sizes=sizes)[0]
                for m, s, b in zip(r["masks"], r["scores"], r["boxes"]):
                    m = m.bool().cpu().numpy()
                    if m.any():
                        out.append(MaskCandidate(m, p, float(s), b.float().cpu().numpy().astype(np.float64)))
        return out


# ---------------------------------------------------------------- Grounding DINO + SAM 2.1
class GroundedSam2Segmenter:
    """Grounding DINO (one joint pass over all prompts) -> per-prompt NMS -> SAM 2.1 box-prompted masks.

    Each detection is assigned to the prompt whose tokens it scores highest on, so prompts compete
    ("bowl" vs "plate") instead of each one grabbing every round thing.
    """

    name = "gdino_sam2"

    def __init__(self, gdino_repo: str | None = None, sam2_repo: str = SAM2_REPO,
                 box_threshold: float = 0.3, nms_iou: float = 0.5, local_only: bool = True):
        self.gdino_repo = gdino_repo
        self.sam2_repo = sam2_repo
        self.box_threshold, self.nms_iou = box_threshold, nms_iou
        self.local_only = local_only
        self._gd = self._gdp = self._sam = self._samp = None
        self._lock = threading.Lock()

    def load(self) -> "GroundedSam2Segmenter":
        with self._lock:
            if self._gd is None:
                from transformers import (GroundingDinoForObjectDetection, GroundingDinoProcessor,
                                          Sam2Model, Sam2Processor)
                dev, dt = _device_dtype()
                t0 = time.time()
                repo = self.gdino_repo or next((r for r in GDINO_REPOS if _is_cached(r)), GDINO_REPOS[-1])
                local = self.local_only and _is_cached(repo)
                self._gdp = GroundingDinoProcessor.from_pretrained(repo, local_files_only=local)
                # Grounding DINO is numerically fragile in bf16; it is small, keep it fp32.
                self._gd = GroundingDinoForObjectDetection.from_pretrained(
                    repo, local_files_only=local).to(dev).eval()
                self.gdino_repo = repo
                local = self.local_only and _is_cached(self.sam2_repo)
                self._samp = Sam2Processor.from_pretrained(self.sam2_repo, local_files_only=local)
                self._sam = Sam2Model.from_pretrained(self.sam2_repo, dtype=dt,
                                                      local_files_only=local).to(dev).eval()
                log.info("Grounding DINO (%s) + SAM 2.1 (%s) loaded on %s in %.1fs",
                         repo, self.sam2_repo, dev, time.time() - t0)
        return self

    # -- detection
    def detect(self, rgb: np.ndarray, prompts: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Boxes (N,4) xyxy px, scores (N,), labels (N,) for the prompts."""
        self.load()
        torch = _torch()
        from PIL import Image
        H, W = rgb.shape[:2]
        prompts = [p.strip().lower() for p in prompts if p.strip()]
        if not prompts:
            return np.zeros((0, 4)), np.zeros(0), []
        text = ". ".join(prompts) + "."
        # Character spans of each prompt in the merged caption -> token indices.
        spans, pos = [], 0
        for p in prompts:
            spans.append((pos, pos + len(p)))
            pos += len(p) + 2
        enc = self._gdp.tokenizer(text, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        tok_of = [[i for i, (a, b) in enumerate(offsets) if b > a and a >= s and b <= e] for s, e in spans]
        dev = self._gd.device
        with torch.inference_mode():
            inputs = self._gdp(images=Image.fromarray(np.ascontiguousarray(rgb)), text=text,
                               return_tensors="pt").to(dev)
            o = self._gd(**inputs)
        probs = o.logits[0].sigmoid().float().cpu().numpy()          # (Q, 256)
        boxes = o.pred_boxes[0].float().cpu().numpy()                  # (Q, 4) cxcywh normalised
        n_tok = inputs["input_ids"].shape[1]
        # Mean over the prompt's tokens (max lets a shared word like "box" claim every boxy thing).
        per_prompt = np.stack([probs[:, [t for t in toks if t < n_tok]].mean(1) if toks else np.zeros(len(probs))
                               for toks in tok_of], axis=1)          # (Q, P)
        lab = per_prompt.argmax(1)
        score = per_prompt.max(1)
        keep = score > self.box_threshold
        cx, cy, w, h = boxes[keep].T
        xyxy = np.stack([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H], 1)
        xyxy = np.clip(xyxy, 0, [W, H, W, H])
        score, lab = score[keep], lab[keep]
        out_b, out_s, out_l = [], [], []
        for k in np.unique(lab):
            idx = np.nonzero(lab == k)[0]
            for j in nms(xyxy[idx], score[idx], self.nms_iou):
                out_b.append(xyxy[idx[j]]); out_s.append(score[idx[j]]); out_l.append(prompts[k])
        if not out_b:
            return np.zeros((0, 4)), np.zeros(0), []
        return np.array(out_b, dtype=np.float64), np.array(out_s, dtype=np.float64), out_l

    def masks_from_boxes(self, rgb: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """SAM 2.1 masks (N, H, W) bool for xyxy boxes."""
        self.load()
        torch = _torch()
        from PIL import Image
        if len(boxes) == 0:
            return np.zeros((0,) + rgb.shape[:2], dtype=bool)
        dev, dt = self._sam.device, self._sam.dtype
        with torch.inference_mode():
            inputs = self._samp(images=Image.fromarray(np.ascontiguousarray(rgb)),
                                input_boxes=[[list(map(float, b)) for b in boxes]], return_tensors="pt").to(dev)
            o = self._sam(pixel_values=inputs["pixel_values"].to(dt), input_boxes=inputs["input_boxes"],
                          multimask_output=False)
            m = self._samp.post_process_masks(o.pred_masks.float().cpu(), inputs["original_sizes"].cpu())[0]
        return m[:, 0].numpy().astype(bool)

    def segment(self, rgb: np.ndarray, prompts: list[str]) -> list[MaskCandidate]:
        boxes, scores, labels = self.detect(rgb, prompts)
        masks = self.masks_from_boxes(rgb, boxes)
        out = []
        for m, s, b, l in zip(masks, scores, boxes, labels):
            # SAM sometimes spills outside the box; clip to a slightly grown box.
            x0, y0, x1, y1 = b
            g = 0.1 * max(x1 - x0, y1 - y0) + 2
            clip = np.zeros_like(m)
            clip[max(int(y0 - g), 0):int(np.ceil(y1 + g)), max(int(x0 - g), 0):int(np.ceil(x1 + g))] = True
            m = m & clip
            if m.any():
                out.append(MaskCandidate(m, l, float(s), b.copy()))
        return out


# ---------------------------------------------------------------- factory
_INSTANCES: dict[str, Segmenter] = {}
_FACTORY_LOCK = threading.Lock()


def make_segmenter(name: str = "auto") -> Segmenter:
    """One loaded segmenter per name per process. "auto" = sam3 if its weights load, else gdino_sam2."""
    with _FACTORY_LOCK:
        if name in _INSTANCES:
            return _INSTANCES[name]
        if name == "sam3":
            seg: Segmenter = Sam3Segmenter().load()
        elif name in ("gdino_sam2", "grounded_sam2"):
            seg = GroundedSam2Segmenter().load()
        elif name == "auto":
            try:
                seg = Sam3Segmenter().load()
                log.info("make_segmenter(auto): using SAM 3")
            except Exception as e:  # weights gated / missing / incompatible
                log.warning("make_segmenter(auto): SAM 3 unavailable (%s: %s); using Grounding DINO + SAM 2.1",
                            type(e).__name__, str(e).splitlines()[0][:200])
                seg = GroundedSam2Segmenter().load()
        else:
            raise ValueError(f"unknown segmenter {name!r} (auto | sam3 | gdino_sam2)")
        _INSTANCES[name] = seg
        _INSTANCES.setdefault(seg.name, seg)
        return seg
