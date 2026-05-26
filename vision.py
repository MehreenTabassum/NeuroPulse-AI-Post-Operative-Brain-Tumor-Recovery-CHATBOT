"""
vision.py
=========
Production 3-D CNN-ViT Hybrid Feature Extractor for Multimodal Neuro-Oncology MRI.

Input  : (B, 4, D, H, W)  — four MRI modalities: T1 · T1CE · T2 · FLAIR
Train  : CNNViTFeatureExtractor.forward(x)          → (B, 2) binary logits
Infer  : CNNViTFeatureExtractor.extract_features(x) → (B, 768) float32 ndarray

Designed to slot into the FastAPI / LangGraph vision microservice:
    features = model.extract_features(tensor)  # np.ndarray (B, 768)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ══════════════════════════════════════════════════════════════════════════════
# §1  Shared building blocks
# ══════════════════════════════════════════════════════════════════════════════

class ConvBnRelu3d(nn.Sequential):
    """Conv3d → BatchNorm3d → ReLU (inplace)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            nn.Conv3d(
                in_ch, out_ch, kernel_size,
                stride=stride, padding=padding,
                dilation=dilation, groups=groups, bias=bias,
            ),
            nn.BatchNorm3d(out_ch, momentum=0.05, eps=1e-5),
            nn.ReLU(inplace=True),
        )


class SEBlock3d(nn.Module):
    """Squeeze-and-Excitation channel attention for 3-D feature maps."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.se(x).view(x.size(0), x.size(1), 1, 1, 1)


# ══════════════════════════════════════════════════════════════════════════════
# §2  CNN Branch  —  Local spatial feature hierarchy
# ══════════════════════════════════════════════════════════════════════════════

class ResBlock3d(nn.Module):
    """
    Pre-activation bottleneck residual block with SE attention.

    in_ch → mid_ch (1x1) → mid_ch (3x3) → out_ch (1x1)

    Optional strided downsampling via the projection shortcut.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        reduction: int = 8,
    ) -> None:
        super().__init__()
        mid_ch = out_ch // 4

        self.bn0  = nn.BatchNorm3d(in_ch,   momentum=0.05)
        self.act0 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(in_ch,   mid_ch, 1, bias=False)
        self.bn1   = nn.BatchNorm3d(mid_ch, momentum=0.05)
        self.act1  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(mid_ch, momentum=0.05)
        self.act2  = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv3d(mid_ch, out_ch, 1, bias=False)
        self.se    = SEBlock3d(out_ch, reduction=reduction)

        self.downsample: nn.Module = (
            nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False)
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.drop = nn.Dropout3d(p=0.04)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x)

        out = self.act0(self.bn0(x))
        out = self.act1(self.bn1(self.conv1(out)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        out = self.se(out)
        out = self.drop(out)

        return out + identity


class CNNBranch(nn.Module):
    """
    3-D CNN backbone producing a compact global feature vector.

    Architecture
    ------------
    Stem  : (B,4,D,H,W)   -> (B,32,D/4,H/4,W/4)   7x7x7 conv, s2 + MaxPool s2

    Stage 0: 32  -> 64   channels, stride 1, 2 ResBlocks
    Stage 1: 64  -> 128  channels, stride 2, 3 ResBlocks   -> /8
    Stage 2: 128 -> 256  channels, stride 2, 4 ResBlocks   -> /16
    Stage 3: 256 -> 512  channels, stride 2, 2 ResBlocks   -> /32

    Head  : AdaptiveAvgPool3d(1) -> Flatten -> Linear(512, cnn_out_dim) -> LN
    """

    def __init__(self, in_ch: int = 4, cnn_out_dim: int = 384) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(32, momentum=0.05),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.stage0 = self._make_stage(32,  64,  n_blocks=2, stride=1)
        self.stage1 = self._make_stage(64,  128, n_blocks=3, stride=2)
        self.stage2 = self._make_stage(128, 256, n_blocks=4, stride=2)
        self.stage3 = self._make_stage(256, 512, n_blocks=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, cnn_out_dim, bias=False),
            nn.LayerNorm(cnn_out_dim),
        )

        self._init_weights()

    @staticmethod
    def _make_stage(
        in_ch: int, out_ch: int, n_blocks: int, stride: int
    ) -> nn.Sequential:
        layers: list[nn.Module] = [ResBlock3d(in_ch, out_ch, stride=stride)]
        for _ in range(1, n_blocks):
            layers.append(ResBlock3d(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.global_pool(x)
        return self.head(x)          # (B, cnn_out_dim)


# ══════════════════════════════════════════════════════════════════════════════
# §3  ViT Branch  —  Global relational features
# ══════════════════════════════════════════════════════════════════════════════

class PatchEmbedding3d(nn.Module):
    """
    Non-overlapping 3-D patch tokeniser via strided Conv3d.

    (B, C, D, H, W) → (B, N_tokens, embed_dim)
    where N_tokens = (D//P) * (H//P) * (W//P)
    """

    def __init__(
        self,
        in_ch: int = 4,
        patch_size: int = 16,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_ch, embed_dim,
            kernel_size=patch_size, stride=patch_size, bias=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int, int]]:
        out = self.proj(x)                    # (B, E, Gd, Gh, Gw)
        B, E, Gd, Gh, Gw = out.shape
        out = out.flatten(2).transpose(1, 2)  # (B, N, E)
        return self.norm(out), (Gd, Gh, Gw)


class FactorisedPositionalEmbedding3d(nn.Module):
    """
    Learnable factorised 3-D positional embeddings.

    Three embedding tables (depth, height, width) are concatenated along the
    feature axis. Avoids a monolithic N*E table; generalises to unseen grid sizes.

        Ed = embed_dim // 3
        Eh = embed_dim // 3
        Ew = embed_dim - 2*(embed_dim//3)   (absorbs remainder)
    """

    def __init__(self, max_grid: int, embed_dim: int) -> None:
        super().__init__()
        self.Ed = embed_dim // 3
        self.Eh = embed_dim // 3
        self.Ew = embed_dim - 2 * (embed_dim // 3)

        self.emb_d = nn.Embedding(max_grid, self.Ed)
        self.emb_h = nn.Embedding(max_grid, self.Eh)
        self.emb_w = nn.Embedding(max_grid, self.Ew)

        nn.init.trunc_normal_(self.emb_d.weight, std=0.02)
        nn.init.trunc_normal_(self.emb_h.weight, std=0.02)
        nn.init.trunc_normal_(self.emb_w.weight, std=0.02)

    def forward(self, grid: Tuple[int, int, int]) -> Tensor:
        Gd, Gh, Gw = grid
        device = self.emb_d.weight.device

        idx_d = torch.arange(Gd, device=device)
        idx_h = torch.arange(Gh, device=device)
        idx_w = torch.arange(Gw, device=device)

        d = self.emb_d(idx_d)[:, None, None, :].expand(Gd, Gh, Gw, self.Ed)
        h = self.emb_h(idx_h)[None, :, None, :].expand(Gd, Gh, Gw, self.Eh)
        w = self.emb_w(idx_w)[None, None, :, :].expand(Gd, Gh, Gw, self.Ew)

        pos = torch.cat([d, h, w], dim=-1)                 # (Gd, Gh, Gw, E)
        return pos.reshape(Gd * Gh * Gw, pos.shape[-1])   # (N, E)


class ViTBranch(nn.Module):
    """
    Vision Transformer encoder for 3-D volumetric MRI.

    Pipeline
    --------
    1. PatchEmbedding3d          -> (B, N, E)
    2. Add CLS token             -> (B, N+1, E)
    3. Add factorised pos embed to token positions (not CLS)
    4. TransformerEncoder (pre-norm, GELU, batch_first)
    5. LayerNorm
    6. Return CLS token          -> (B, E)
    """

    def __init__(
        self,
        in_ch: int = 4,
        patch_size: int = 16,
        input_size: int = 128,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert input_size % patch_size == 0, (
            f"input_size ({input_size}) must be divisible by patch_size ({patch_size})"
        )
        self.max_grid = input_size // patch_size

        self.patch_embed = PatchEmbedding3d(in_ch, patch_size, embed_dim)
        self.pos_embed   = FactorisedPositionalEmbedding3d(self.max_grid, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_drop = nn.Dropout(p=proj_dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=proj_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]

        tokens, grid = self.patch_embed(x)          # (B, N, E)
        pos          = self.pos_embed(grid)          # (N, E)
        tokens       = tokens + pos.unsqueeze(0)    # broadcast over batch

        cls    = self.cls_token.expand(B, -1, -1)   # (B, 1, E)
        tokens = torch.cat([cls, tokens], dim=1)    # (B, N+1, E)
        tokens = self.pos_drop(tokens)

        tokens = self.transformer(tokens)            # (B, N+1, E)
        tokens = self.norm(tokens)

        return tokens[:, 0]                          # (B, E)  -- CLS token


# ══════════════════════════════════════════════════════════════════════════════
# §4  Fusion Module  —  Bidirectional cross-attention + MLP to 768-d
# ══════════════════════════════════════════════════════════════════════════════

class CrossAttentionFusion(nn.Module):
    """
    Bidirectional single-head cross-attention fusion of CNN and ViT vectors.

    CNN feat queries ViT context;  ViT feat queries CNN context.
    Both attended outputs are concatenated with original features then
    projected through a two-layer MLP to `out_dim` (768).

    cnn_feat : (B, cnn_dim)
    vit_feat : (B, vit_dim)
    output   : (B, out_dim)
    """

    def __init__(
        self,
        cnn_dim: int,
        vit_dim: int,
        out_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        inner = min(cnn_dim, vit_dim)
        self.scale = math.sqrt(inner)

        self.q_cnn = nn.Linear(cnn_dim, inner, bias=False)
        self.k_vit = nn.Linear(vit_dim, inner, bias=False)
        self.v_vit = nn.Linear(vit_dim, inner, bias=False)

        self.q_vit = nn.Linear(vit_dim, inner, bias=False)
        self.k_cnn = nn.Linear(cnn_dim, inner, bias=False)
        self.v_cnn = nn.Linear(cnn_dim, inner, bias=False)

        concat_dim = cnn_dim + inner + vit_dim + inner
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, out_dim * 2, bias=False),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim * 2, out_dim, bias=False),
            nn.LayerNorm(out_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _attend(q: Tensor, k: Tensor, v: Tensor, scale: float) -> Tensor:
        # q, k, v: (B, inner) — compute scaled dot-product attention score per sample
        # (B, inner) * (B, inner) -> (B,) dot product, then weight v
        attn = F.softmax((q * k).sum(dim=-1, keepdim=True) / scale, dim=0)  # (B, 1)
        return attn * v  # (B, inner)

    def forward(self, cnn_feat: Tensor, vit_feat: Tensor) -> Tensor:
        attn_c2v = self._attend(
            self.q_cnn(cnn_feat), self.k_vit(vit_feat), self.v_vit(vit_feat), self.scale
        )
        attn_v2c = self._attend(
            self.q_vit(vit_feat), self.k_cnn(cnn_feat), self.v_cnn(cnn_feat), self.scale
        )
        fused = torch.cat([cnn_feat, attn_c2v, vit_feat, attn_v2c], dim=-1)
        return self.mlp(fused)      # (B, 768)


# ══════════════════════════════════════════════════════════════════════════════
# §5  Top-level model
# ══════════════════════════════════════════════════════════════════════════════

class CNNViTFeatureExtractor(nn.Module):
    """
    3-D CNN-ViT Hybrid Feature Extractor for Multimodal MRI.

    Parameters
    ----------
    in_channels  : MRI modalities (default 4: T1, T1CE, T2, FLAIR)
    cnn_dim      : CNN branch output channels           (default 384)
    vit_dim      : ViT embedding dimension              (default 384)
    feature_dim  : unified latent dimension             (default 768)
    num_classes  : classifier outputs                   (default 2)
    patch_size   : ViT patch edge length                (default 16)
    input_size   : isotropic voxel resolution           (default 128)
    vit_depth    : Transformer encoder layers           (default 8)
    vit_heads    : multi-head attention heads           (default 6)
    mlp_ratio    : ViT MLP hidden expansion             (default 4.0)
    dropout      : ViT and classifier dropout rate      (default 0.1)

    Training
    --------
        model.train()
        logits = model(x)                       # (B, 2)
        loss   = F.cross_entropy(logits, labels)

    Inference / FastAPI tool
    ------------------------
        model.eval()
        feats = model.extract_features(x)       # np.ndarray (B, 768) float32
        return feats[0].tolist()                 # list[float] for JSON response
    """

    def __init__(
        self,
        in_channels: int = 4,
        cnn_dim: int = 384,
        vit_dim: int = 384,
        feature_dim: int = 768,
        num_classes: int = 2,
        patch_size: int = 16,
        input_size: int = 128,
        vit_depth: int = 8,
        vit_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.cnn_branch = CNNBranch(in_ch=in_channels, cnn_out_dim=cnn_dim)

        self.vit_branch = ViTBranch(
            in_ch=in_channels,
            patch_size=patch_size,
            input_size=input_size,
            embed_dim=vit_dim,
            depth=vit_depth,
            num_heads=vit_heads,
            mlp_ratio=mlp_ratio,
            proj_dropout=dropout,
        )

        self.fusion = CrossAttentionFusion(
            cnn_dim=cnn_dim,
            vit_dim=vit_dim,
            out_dim=feature_dim,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, feature_dim // 2, bias=False),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(feature_dim // 2, num_classes, bias=True),
        )

        self._init_classifier()

    def _init_classifier(self) -> None:
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Shared encoder ────────────────────────────────────────────────────────

    def _encode(self, x: Tensor) -> Tensor:
        """Run both branches + fusion → (B, 768) latent vector."""
        cnn_feat: Tensor = self.cnn_branch(x)
        vit_feat: Tensor = self.vit_branch(x)
        return self.fusion(cnn_feat, vit_feat)

    # ── Forward — training ────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, 4, D, H, W) float32
        Returns:
            logits: (B, num_classes)
        """
        return self.classifier(self._encode(x))

    # ── Feature extraction — FastAPI / LangGraph ──────────────────────────────

    @torch.inference_mode()
    def extract_features(
        self,
        x: Tensor,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """
        Extract the raw 768-d embedding without the classifier head.

        Thread-safe: torch.inference_mode() prevents gradient accumulation.
        Handles eval/train mode toggling automatically.

        Args:
            x      : (B, 4, D, H, W) float32 — z-score normalised MRI volume.
            device : target device; inferred from model parameters if None.

        Returns:
            np.ndarray shape (B, 768), dtype float32, contiguous, CPU.
        """
        if device is None:
            try:
                device = next(self.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        x = x.to(device=device, dtype=torch.float32)

        was_training = self.training
        self.eval()

        z: Tensor = self._encode(x)

        if was_training:
            self.train()

        return z.detach().cpu().numpy().astype(np.float32)

    # ── Convenience ───────────────────────────────────────────────────────────

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════════
# §6  Factory helper
# ══════════════════════════════════════════════════════════════════════════════

def build_model(
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    **kwargs,
) -> CNNViTFeatureExtractor:
    """
    Instantiate, optionally restore weights, and move to device.

    Args:
        weights_path : path to .pt checkpoint (raw state_dict or
                       {"state_dict": ...} wrapper). None = random init.
        device       : auto-selects CUDA if available when None.
        **kwargs     : forwarded to CNNViTFeatureExtractor.__init__.

    Returns:
        model in eval() mode on device.

    Example:
        model = build_model(
            weights_path=os.getenv("WEIGHTS_PATH"),
            device=torch.device("cuda"),
        )
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNViTFeatureExtractor(**kwargs).to(device)

    if weights_path is not None:
        import os
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
        ckpt = torch.load(weights_path, map_location=device)
        state_dict = ckpt.get("state_dict", ckpt)
        missing, unexpected = model.load_state_dict(state_dict, strict=True)
        if missing:
            raise RuntimeError(f"Missing keys in checkpoint: {missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")

    model.eval()
    return model
