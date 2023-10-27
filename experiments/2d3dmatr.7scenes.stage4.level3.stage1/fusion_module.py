from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from vision3d.layers import FourierEmbedding, TransformerLayer


class CrossModalFusionModule(nn.Module):
    def __init__(
        self,
        img_input_dim: int,
        pcd_input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_heads: int,
        blocks: List[str],
        dropout: Optional[float] = None,
        activation_fn: str = "ReLU",
        use_embedding: bool = True,
        embedding_dim: int = 10,
    ):
        super().__init__()

        self.use_embedding = use_embedding
        if self.use_embedding:
            self.embedding = FourierEmbedding(embedding_dim, use_pi=False, use_input=True)
            self.img_emb_proj = nn.Linear(embedding_dim * 4 + 2, hidden_dim)
            self.pcd_emb_proj = nn.Linear(embedding_dim * 6 + 3, hidden_dim)
        else:
            self.embedding = None
            self.img_emb_proj = None
            self.pcd_emb_proj = None

        self.img_in_proj = nn.Linear(img_input_dim, hidden_dim)
        self.pcd_in_proj = nn.Linear(pcd_input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        self.blocks = blocks
        layers = []
        for block in self.blocks:
            assert block in ["self", "cross"]
            layers.append(TransformerLayer(hidden_dim, num_heads, dropout=dropout, act_cfg=activation_fn))
        self.transformer = nn.ModuleList(layers)

    def create_2d_embedding(self, pixels):
        embeddings = self.embedding(pixels)  # (1, HxW, L)
        embeddings = self.img_emb_proj(embeddings)  # (1, HxW, C)
        return embeddings

    def create_3d_embedding(self, points):
        points = points - points.mean(dim=1)
        embeddings = self.embedding(points)
        embeddings = self.pcd_emb_proj(embeddings)
        return embeddings

    def forward(
        self,
        img_feats: Tensor,
        img_pixels: Tensor,
        pcd_feats: Tensor,
        pcd_points: Tensor,
        img_masks: Optional[Tensor] = None,
        pcd_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Cross-Modal Feature Fusion Module.

        Args:
            img_feats (tensor): the features of the image in the shape of (B, HxW, Ci).
            img_pixels (tensor): the coordinates of the image in the shape of (B, HxW, 2).
            pcd_feats (tensor): the features of the point cloud in the shape of (B, N, Cp).
            pcd_points (tensor): the coordinates of the point cloud in the shape of (B, N, 3).
            img_masks (tensor, optional): the masks of the image in the shape of (B, H, W).
            pcd_masks (tensor, optional): the masks of the point cloud in the shape of (B, N).

        Returns:
            A tensor of the fused features of the image in the shape of (B, Co, H, W).
            A tensor of the fused features of the point cloud in the shape of (N, Co).
        """
        img_tokens = self.img_in_proj(img_feats)  # (B, HxW, Ci) -> (B, HxW, C)
        pcd_tokens = self.pcd_in_proj(pcd_feats)  # (B, N, Cp) -> (B, N, C)

        if self.use_embedding:
            img_embeddings = self.create_2d_embedding(img_pixels)  # (B, HxW, C)
            img_tokens = img_tokens + img_embeddings  # (B, HxW, C)
            pcd_embeddings = self.create_3d_embedding(pcd_points)  # (B, N, C)
            pcd_tokens = pcd_tokens + pcd_embeddings  # (B, N, C)

        for i, block in enumerate(self.blocks):
            if block == "self":
                img_tokens = self.transformer[i](img_tokens, img_tokens, img_tokens, k_masks=img_masks)
                pcd_tokens = self.transformer[i](pcd_tokens, pcd_tokens, pcd_tokens, k_masks=pcd_masks)
            else:
                img_tokens = self.transformer[i](img_tokens, pcd_tokens, pcd_tokens, k_masks=pcd_masks)
                pcd_tokens = self.transformer[i](pcd_tokens, img_tokens, img_tokens, k_masks=img_masks)

        img_feats = self.out_proj(img_tokens)  # (B, HxW, C)
        pcd_feats = self.out_proj(pcd_tokens)  # (B, N, C)

        return img_feats, pcd_feats
