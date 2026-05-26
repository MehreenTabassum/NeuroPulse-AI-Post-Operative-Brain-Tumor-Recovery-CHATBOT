"""
vision_service.py — CNN-ViT Vision Microservice
Post-Operative Brain Tumor Recovery Analysis · BraTS-2024
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import tempfile
import time
import uuid
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vision import build_model, CNNViTFeatureExtractor as TorchModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

FEATURE_DIM: int = 768
ALLOWED_SUFFIXES: tuple[str, ...] = (".nii", ".nii.gz")
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "512"))
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024
TARGET_SHAPE: tuple[int, int, int] = (128, 128, 128)


class CNNViTFeatureExtractor:
    """
    Thin wrapper around TorchModel (vision.py).
    Gracefully falls back to a random feature vector if the model
    cannot be loaded due to insufficient RAM.
    """

    _instance: "CNNViTFeatureExtractor | None" = None

    def __init__(self, weights_path: str | None = None) -> None:
        self.weights_path  = weights_path
        self.feature_dim   = FEATURE_DIM
        self.is_real_model = False
        self._model: TorchModel | None = None
        self._device       = torch.device("cpu")  # force CPU — safest for low RAM
        self._load_model()

    def _load_model(self) -> None:
        """Try to load a minimal CNN-ViT; fall back silently on OOM."""
        configs = [
            # Try progressively smaller models until one fits in RAM
            dict(vit_depth=2, vit_heads=2, cnn_dim=64,  vit_dim=64),
            dict(vit_depth=1, vit_heads=1, cnn_dim=32,  vit_dim=32),
            dict(vit_depth=1, vit_heads=1, cnn_dim=16,  vit_dim=16),
        ]
        for cfg in configs:
            try:
                self._model = build_model(
                    weights_path=self.weights_path,
                    device=self._device,
                    **cfg,
                )
                self.is_real_model = True
                logger.info("✔ CNN-ViT loaded | config=%s | device=%s", cfg, self._device)
                return
            except Exception as exc:
                logger.warning("CNN-ViT config %s failed: %s — trying smaller...", cfg, exc)

        # All configs failed — use random fallback (warns in response)
        logger.error(
            "All CNN-ViT configs failed (likely OOM). "
            "Using random feature vectors. Free more RAM for real features."
        )
        self._model = None
        self.is_real_model = False

    @staticmethod
    def _preprocess(volume: np.ndarray) -> np.ndarray:
        def _crop_or_pad(arr: np.ndarray, axis: int, target: int) -> np.ndarray:
            size = arr.shape[axis]
            if size >= target:
                start = (size - target) // 2
                slc = [slice(None)] * arr.ndim
                slc[axis] = slice(start, start + target)
                return arr[tuple(slc)]
            pad_before = (target - size) // 2
            pad_after  = target - size - pad_before
            pad_width  = [(0, 0)] * arr.ndim
            pad_width[axis] = (pad_before, pad_after)
            return np.pad(arr, pad_width, mode="constant", constant_values=0)

        vol = volume.astype(np.float32)
        for axis, target in enumerate(TARGET_SHAPE):
            vol = _crop_or_pad(vol, axis, target)

        mask = vol > 0
        if mask.sum() > 0:
            mean = vol[mask].mean()
            std  = vol[mask].std() + 1e-8
            vol  = np.where(mask, (vol - mean) / std, 0.0)
        return vol

    def extract(self, volume: np.ndarray) -> list[float]:
        # Fallback: model couldn't load due to RAM
        if self._model is None:
            logger.warning("Using random feature vector (model not loaded).")
            rng = random.Random(int(np.abs(volume).sum()) % (2**32))
            return [rng.gauss(0, 0.1) for _ in range(FEATURE_DIM)]

        preprocessed = self._preprocess(volume)
        tensor = torch.from_numpy(preprocessed).unsqueeze(0)   # (1,128,128,128)
        tensor = tensor.unsqueeze(0).repeat(1, 4, 1, 1, 1)    # (1,4,128,128,128)

        logger.info("CNN-ViT forward pass | input=%s | device=%s",
                    tuple(tensor.shape), self._device)

        features: np.ndarray = self._model.extract_features(tensor, device=self._device)
        return features[0].tolist()

    @classmethod
    def get_instance(cls, weights_path: str | None = None) -> "CNNViTFeatureExtractor":
        if cls._instance is None:
            cls._instance = cls(weights_path=weights_path)
        return cls._instance


# ─── Pydantic models ──────────────────────────────────────────────────────────

class FeatureExtractionResponse(BaseModel):
    request_id:        str         = Field(...)
    filename:          str         = Field(...)
    volume_shape:      list[int]   = Field(...)
    voxel_size_mm:     list[float] = Field(...)
    feature_dim:       int         = Field(...)
    features:          list[float] = Field(...)
    volume_hash:       str         = Field(...)
    is_stub_model:     bool        = Field(...)
    processing_time_ms: float      = Field(...)


class ModelInfoResponse(BaseModel):
    architecture:    str
    feature_dim:     int
    target_shape:    list[int]
    weights_loaded:  bool
    allowed_formats: list[str]
    max_upload_mb:   int
    device:          str


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="CNN-ViT Vision Microservice",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _global_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {exc}"},
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _validate_filename(filename: str) -> None:
    if not any(filename.lower().endswith(s) for s in ALLOWED_SUFFIXES):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Accepted formats: {', '.join(ALLOWED_SUFFIXES)}.",
        )


def _load_nifti_from_bytes(raw: bytes, original_name: str) -> tuple[np.ndarray, list[float]]:
    suffix = ".nii.gz" if original_name.lower().endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name
    try:
        img = nib.load(tmp_path)
        volume = np.array(img.dataobj, dtype=np.float32)  # load into memory FIRST
        try:
            voxel_size = [float(z) for z in img.header.get_zooms()[:3]]
        except Exception:
            voxel_size = [1.0, 1.0, 1.0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse NIfTI: {exc}") from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)  # delete AFTER data is in memory

    if volume.ndim < 3:
        raise HTTPException(status_code=400, detail=f"Expected 3-D volume, got {volume.shape}.")
    if volume.ndim == 4:
        volume = volume[..., 0]

    logger.info("NIfTI loaded | shape=%s | voxel_size_mm=%s", volume.shape, voxel_size)
    return volume, voxel_size


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "cnn-vit-vision"}


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
async def model_info() -> ModelInfoResponse:
    extractor = CNNViTFeatureExtractor.get_instance(weights_path=os.getenv("WEIGHTS_PATH"))
    return ModelInfoResponse(
        architecture="CNN-ViT (3D ResNet + ViT encoder, vision.py)",
        feature_dim=extractor.feature_dim,
        target_shape=list(TARGET_SHAPE),
        weights_loaded=extractor.is_real_model,
        allowed_formats=list(ALLOWED_SUFFIXES),
        max_upload_mb=MAX_UPLOAD_MB,
        device=str(extractor._device),
    )


@app.post("/extract-features", response_model=FeatureExtractionResponse,
          status_code=status.HTTP_200_OK, tags=["vision"])
async def extract_features(
    file: UploadFile = File(..., description="3-D MRI scan (.nii or .nii.gz)."),
) -> FeatureExtractionResponse:
    request_id = str(uuid.uuid4())
    filename   = file.filename or "upload.nii.gz"
    logger.info("▶ /extract-features | request_id=%s | file=%s", request_id, filename)

    _validate_filename(filename)
    raw: bytes = await file.read()

    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400,
            detail=f"Upload exceeds {MAX_UPLOAD_MB} MB ({len(raw)/1024/1024:.1f} MB received).")

    volume, voxel_size = _load_nifti_from_bytes(raw, filename)
    volume_hash = hashlib.md5(volume.tobytes()).hexdigest()

    extractor = CNNViTFeatureExtractor.get_instance(weights_path=os.getenv("WEIGHTS_PATH"))

    t0 = time.perf_counter()
    features = extractor.extract(volume)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    assert len(features) == FEATURE_DIM, f"Expected {FEATURE_DIM} dims, got {len(features)}."

    logger.info("✔ /extract-features | request_id=%s | elapsed_ms=%.2f | real_model=%s",
                request_id, elapsed_ms, extractor.is_real_model)

    return FeatureExtractionResponse(
        request_id=request_id,
        filename=filename,
        volume_shape=list(volume.shape),
        voxel_size_mm=voxel_size,
        feature_dim=FEATURE_DIM,
        features=features,
        volume_hash=volume_hash,
        is_stub_model=not extractor.is_real_model,
        processing_time_ms=elapsed_ms,
    )


if __name__ == "__main__":
    uvicorn.run(
        "vision_service:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8001")),
        reload=False,
        log_level="info",
    )
