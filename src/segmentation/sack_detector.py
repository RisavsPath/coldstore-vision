"""YOLO instance-segmentation wrapper: sack / rack_beam / empty_slot polygons + masks.

Uses the trained model at `detector.weights` (models/weights/best.pt) when present. Until training
finishes it falls back to the open-vocabulary YOLOE model prompted with `detector.fallback_prompts`,
so the downstream pipeline always runs on YOLO instance masks. Dropping best.pt into
models/weights/ switches over automatically; nothing else changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.config import resolve

CLASSES = ("sack", "rack_beam", "empty_slot")


@dataclass
class Instance:
    cls: str
    conf: float
    box: np.ndarray            # x1, y1, x2, y2 (float, pixels)
    mask: np.ndarray           # HxW bool
    polygon: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    @property
    def centroid(self) -> tuple[float, float]:
        ys, xs = np.nonzero(self.mask)
        if len(xs) == 0:
            return float((self.box[0] + self.box[2]) / 2), float((self.box[1] + self.box[3]) / 2)
        return float(xs.mean()), float(ys.mean())


def mask_to_polygon(mask: np.ndarray) -> np.ndarray:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros((0, 2), np.float32)
    return max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)


def instance_from_mask(cls: str, mask: np.ndarray, conf: float = 1.0) -> Instance | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32)
    return Instance(cls, conf, box, mask.astype(bool), mask_to_polygon(mask))


class SackDetector:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO

        d = cfg["detector"]
        self.conf, self.iou, self.imgsz, self.device = d["conf"], d["iou"], d["imgsz"], d.get("device", "cpu")
        weights = resolve(d["weights"])
        self.using_trained = weights.exists()
        if self.using_trained:
            self.model = YOLO(str(weights))
            self.class_map = {i: n for i, n in self.model.names.items()}
        else:
            import contextlib

            from ultralytics import YOLOE

            hub = resolve("models/hub")
            hub.mkdir(parents=True, exist_ok=True)
            with contextlib.chdir(hub):  # YOLOE + its MobileCLIP text encoder download/load from the CWD
                self.model = YOLOE(d["fallback_weights"])
                prompts, self.class_map = [], {}
                for cls, plist in d["fallback_prompts"].items():
                    for p in plist:
                        self.class_map[len(prompts)] = cls
                        prompts.append(p)
                self.model.set_classes(prompts, self.model.get_text_pe(prompts))
        self.source = str(weights) if self.using_trained else f"{d['fallback_weights']} (fallback)"

    def detect(self, img_bgr: np.ndarray) -> list[Instance]:
        res = self.model.predict(img_bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz, device=self.device,
                                 retina_masks=True, agnostic_nms=not self.using_trained, verbose=False)[0]
        out: list[Instance] = []
        if res.masks is None or res.boxes is None:
            return out
        h, w = img_bgr.shape[:2]
        masks = res.masks.data.cpu().numpy()
        for k in range(len(res.boxes)):
            cls = self.class_map.get(int(res.boxes.cls[k]), "sack")
            m = masks[k]
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            m = m > 0.5
            if m.sum() < 50:
                continue
            out.append(Instance(cls, float(res.boxes.conf[k]), res.boxes.xyxy[k].cpu().numpy(), m,
                                res.masks.xy[k].astype(np.float32)))
        return out
