"""
Module 01 – Image Preprocessing
────────────────────────────────────────────────────────────────────────────────
Normalises raw traffic images before they enter the detection pipeline.

Pipeline:
  1. CLAHE  – local contrast normalisation for low-light scenes
  2. Retinex illumination correction (single-scale)
  3. Bilateral denoising   – edge-preserving noise removal
  4. Gamma correction      – shadow / overexposure compensation
  5. Resize + letterbox    – consistent input size for YOLO
  6. Channel normalisation – ImageNet mean/std subtraction
  7. Metadata embedding    – camera ID, timestamp, GPS injected into result
"""

import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    camera_id: str = "unknown"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    gps_lat: Optional[float] = None
    gps_lon: Optional[float] = None
    original_size: Tuple[int, int] = (0, 0)   # (H, W)
    processed_size: Tuple[int, int] = (640, 640)
    capture_conditions: str = "unknown"


@dataclass
class PreprocessedImage:
    image: np.ndarray           # float32 normalised tensor-ready array
    image_display: np.ndarray   # uint8 BGR, annotatable copy
    metadata: ImageMetadata
    enhancement_log: Dict[str, Any] = field(default_factory=dict)


class ImagePreprocessor:
    """Applies the full preprocessing pipeline to a single traffic image."""

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        pp = cfg["preprocessing"]

        self.target_size: Tuple[int, int] = tuple(pp["target_size"])   # (W, H)
        self.clahe = cv2.createCLAHE(
            clipLimit=pp["clahe"]["clip_limit"],
            tileGridSize=tuple(pp["clahe"]["tile_grid_size"]),
        )
        self.denoise_enabled: bool = pp["denoise"]["enabled"]
        self.denoise_h: int = pp["denoise"]["h"]
        self.gamma: float = pp["gamma_correction"]
        self.mean = np.array(pp["normalize"]["mean"], dtype=np.float32)
        self.std  = np.array(pp["normalize"]["std"],  dtype=np.float32)

        # Pre-build gamma LUT
        self._gamma_lut = self._build_gamma_lut(self.gamma)

    # ── Public API ───────────────────────────────────────────────────────────

    def process(
        self,
        image_input: "str | Path | np.ndarray",
        metadata: Optional[ImageMetadata] = None,
    ) -> PreprocessedImage:
        """
        Run the full preprocessing pipeline.

        Parameters
        ----------
        image_input : path (str/Path) or BGR numpy array
        metadata    : optional pre-populated ImageMetadata

        Returns
        -------
        PreprocessedImage with normalised tensor and display copy
        """
        img_bgr = self._load(image_input)
        if metadata is None:
            metadata = ImageMetadata()
        metadata.original_size = img_bgr.shape[:2]  # (H, W)

        log: Dict[str, Any] = {}

        # 1. CLAHE
        img_bgr, log["clahe"] = self._apply_clahe(img_bgr)

        # 2. Retinex
        img_bgr, log["retinex"] = self._apply_retinex(img_bgr)

        # 3. Denoise
        if self.denoise_enabled:
            img_bgr, log["denoise"] = self._apply_denoise(img_bgr)

        # 4. Gamma correction
        img_bgr, log["gamma"] = self._apply_gamma(img_bgr)

        # 5. Letterbox resize
        img_resized, log["resize"] = self._letterbox(img_bgr, self.target_size)
        metadata.processed_size = self.target_size

        # Keep display copy (uint8) before normalisation
        img_display = img_resized.copy()

        # 6. Normalise to float32 [0, 1] then ImageNet-subtract
        img_norm = self._normalize(img_resized)

        logger.debug("Preprocessing complete | %s", log)
        return PreprocessedImage(
            image=img_norm,
            image_display=img_display,
            metadata=metadata,
            enhancement_log=log,
        )

    def process_batch(
        self,
        images: list,
        metadatas: Optional[list] = None,
    ) -> list:
        """Process a list of images. Returns list of PreprocessedImage."""
        if metadatas is None:
            metadatas = [None] * len(images)
        return [self.process(img, meta) for img, meta in zip(images, metadatas)]

    # ── Enhancement steps ────────────────────────────────────────────────────

    def _apply_clahe(self, img: np.ndarray) -> Tuple[np.ndarray, str]:
        """CLAHE on L channel of LAB colour space."""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR), "applied"

    def _apply_retinex(self, img: np.ndarray, sigma: float = 300.0) -> Tuple[np.ndarray, str]:
        """Single-Scale Retinex: separates reflectance from illumination."""
        img_f = img.astype(np.float32) + 1.0
        blurred = cv2.GaussianBlur(img_f, (0, 0), sigma)
        retinex = np.log10(img_f) - np.log10(blurred + 1.0)
        # Rescale to [0, 255]
        retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
        return retinex.astype(np.uint8), "applied"

    def _apply_denoise(self, img: np.ndarray) -> Tuple[np.ndarray, str]:
        """Fast bilateral filter (edge-preserving)."""
        denoised = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        return denoised, f"bilateral h={self.denoise_h}"

    def _apply_gamma(self, img: np.ndarray) -> Tuple[np.ndarray, str]:
        """Gamma correction via pre-built LUT."""
        return cv2.LUT(img, self._gamma_lut), f"gamma={self.gamma}"

    def _letterbox(
        self,
        img: np.ndarray,
        target: Tuple[int, int],
        colour: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, Dict]:
        """Resize with aspect-ratio preservation and grey padding."""
        h, w = img.shape[:2]
        tw, th = target
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((th, tw, 3), colour, dtype=np.uint8)
        pad_x = (tw - nw) // 2
        pad_y = (th - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = img_resized
        return canvas, {"scale": scale, "pad": (pad_x, pad_y), "resized": (nw, nh)}

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Convert BGR uint8 → RGB float32, subtract ImageNet stats."""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return (rgb - self.mean) / self.std

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load(source: "str | Path | np.ndarray") -> np.ndarray:
        if isinstance(source, np.ndarray):
            return source.copy()
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Could not decode image: {path}")
        return img

    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        inv_gamma = 1.0 / gamma
        table = np.array(
            [(i / 255.0) ** inv_gamma * 255 for i in range(256)],
            dtype=np.uint8,
        )
        return table
