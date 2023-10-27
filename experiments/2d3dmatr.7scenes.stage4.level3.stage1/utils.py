from typing import Tuple

import torch
from torch import Tensor

from vision3d.ops import apply_transform, index_select, knn, masked_mean, pairwise_distance


def batchify(knn_inputs, block_h, block_w, stride):
    squeeze_last = False
    if knn_inputs.dim() == 2:
        knn_inputs = knn_inputs.unsqueeze(-1)
        squeeze_last = True

    num_inputs, num_neighbors, num_channels = knn_inputs.shape
    assert num_neighbors == block_h * block_w
    knn_inputs = knn_inputs.view(num_inputs, block_h // stride, stride, block_w // stride, stride, num_channels)
    knn_inputs = knn_inputs.permute(0, 2, 4, 1, 3, 5).contiguous()
    knn_inputs = knn_inputs.view(num_inputs * stride * stride, (block_h // stride) * (block_w // stride), num_channels)

    if squeeze_last:
        knn_inputs = knn_inputs.squeeze(-1)

    return knn_inputs


@torch.no_grad()
def patchify(
    img_points: Tensor,
    img_pixels: Tensor,
    img_masks: Tensor,
    img_h_f: int,
    img_w_f: int,
    img_h_c: int,
    img_w_c: int,
    stride: int = 1,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    assert img_h_f % img_h_c == 0, f"Image height must be divisible by patch height ({img_h_f} vs {img_h_c})."
    assert img_w_f % img_w_c == 0, f"Image width must be divisible by patch width ({img_w_f} vs {img_w_c})."
    indices = torch.arange(img_h_f * img_w_f).cuda().view(img_h_f, img_w_f)
    knn_indices = indices.view(img_h_c, img_h_f // img_h_c, img_w_c, img_w_f // img_w_c)  # (H', H/H', W', W/W')
    knn_indices = knn_indices.permute(0, 2, 1, 3).contiguous()  # (H', W', H/H', W/W')
    if stride > 1:
        knn_indices = knn_indices[..., ::stride, ::stride].contiguous()  # (H', W', H/H'/S, W/W'/S)
    knn_indices = knn_indices.view(img_h_c * img_w_c, -1)  # (H'xW', BhxBw)
    knn_points = index_select(img_points, knn_indices, dim=0)
    knn_pixels = index_select(img_pixels, knn_indices, dim=0)
    knn_masks = index_select(img_masks, knn_indices, dim=0)
    masks = torch.any(knn_masks, dim=1)
    return knn_points, knn_pixels, knn_indices, knn_masks, masks


@torch.no_grad()
def get_2d3d_node_correspondences(
    img_masks: Tensor,
    img_knn_points: Tensor,
    img_knn_pixels: Tensor,
    img_knn_masks: Tensor,
    pcd_masks: Tensor,
    pcd_knn_points: Tensor,
    pcd_knn_pixels: Tensor,
    pcd_knn_masks: Tensor,
    transform: Tensor,
    pos_radius_2d: float,
    pos_radius_3d: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate 2D-3D ground-truth superpoint/patch correspondences.

    Each patch is composed of at most k-nearest points of the corresponding superpoint.
    A pair of points match if their 3D distance is below `pos_radius_3d` AND their 2D distance is below `pos_radius_2d`.

    Args:
        img_masks (tensor[bool]): (M,)
        img_knn_points (tensor): (M, Ki, 3)
        img_knn_pixels (tensor): (M, Ki, 2)
        img_knn_masks (tensor[bool]): (M, Ki)
        pcd_masks (tensor[bool]): (N,)
        pcd_knn_points (tensor): (N, Kc, 3)
        pcd_knn_pixels (tensor): (N, Kc, 3)
        pcd_knn_masks (tensor[bool]): (N, Kc)
        transform (tensor): (4, 4)
        pos_radius_2d (float)
        pos_radius_3d (float)

    Returns:
        src_corr_indices (LongTensor): (C,)
        tgt_corr_indices (LongTensor): (C,)
        corr_overlaps (Tensor): (C,)
    """
    pcd_knn_points = apply_transform(pcd_knn_points, transform)  # (N, Kc, 3)

    img_centers = masked_mean(img_knn_points, img_knn_masks)  # (M, 3)
    pcd_centers = masked_mean(pcd_knn_points, pcd_knn_masks)  # (N, 3)

    # filter out non-overlapping patches using enclosing sphere
    img_knn_dists = torch.linalg.norm(img_knn_points - img_centers.unsqueeze(1), dim=-1)  # (M, K)
    img_knn_dists[~img_knn_masks] = 0.0
    img_max_dists = img_knn_dists.max(1)[0]  # (M,)
    pcd_knn_dists = torch.linalg.norm(pcd_knn_points - pcd_centers.unsqueeze(1), dim=-1)  # (N, K)
    pcd_knn_dists[~pcd_knn_masks] = 0.0
    pcd_max_dists = pcd_knn_dists.max(1)[0]  # (N,)
    dist_mat = torch.sqrt(pairwise_distance(img_centers, pcd_centers))  # (M, N)
    intersect_mat = torch.gt(img_max_dists.unsqueeze(1) + pcd_max_dists.unsqueeze(0) + pos_radius_3d - dist_mat, 0.0)
    intersect_mat = torch.logical_and(intersect_mat, img_masks.unsqueeze(1))
    intersect_mat = torch.logical_and(intersect_mat, pcd_masks.unsqueeze(0))
    candidate_img_indices, candidate_pcd_indices = torch.nonzero(intersect_mat, as_tuple=True)

    num_candidates = candidate_img_indices.shape[0]

    # select potential patch pairs, compute correspondence matrix
    img_knn_points = img_knn_points[candidate_img_indices]  # (B, Ki, 3)
    img_knn_pixels = img_knn_pixels[candidate_img_indices]  # (B, Ki, 2)
    img_knn_masks = img_knn_masks[candidate_img_indices]  # (B, Ki)
    pcd_knn_points = pcd_knn_points[candidate_pcd_indices]  # (B, Kc, 3)
    pcd_knn_pixels = pcd_knn_pixels[candidate_pcd_indices]  # (B, Kc, 2)
    pcd_knn_masks = pcd_knn_masks[candidate_pcd_indices]  # (B, Ki)

    # compute 2d overlap masks
    img_knn_min_distances_3d, img_knn_min_indices_3d = knn(img_knn_points, pcd_knn_points, k=1, return_distance=True)
    img_knn_min_distances_3d = img_knn_min_distances_3d.squeeze(-1)
    img_knn_min_indices_3d = img_knn_min_indices_3d.squeeze(-1)
    img_knn_batch_indices_3d = torch.arange(num_candidates).cuda().unsqueeze(1).expand_as(img_knn_min_indices_3d)
    img_knn_min_pcd_pixels = pcd_knn_pixels[img_knn_batch_indices_3d, img_knn_min_indices_3d]
    img_knn_min_distances_2d = torch.linalg.norm(img_knn_pixels - img_knn_min_pcd_pixels, dim=-1)
    img_knn_min_pcd_masks = pcd_knn_masks[img_knn_batch_indices_3d, img_knn_min_indices_3d]
    img_knn_overlap_masks_3d = torch.lt(img_knn_min_distances_3d, pos_radius_3d)
    img_knn_overlap_masks_2d = torch.lt(img_knn_min_distances_2d, pos_radius_2d)
    img_knn_overlap_masks = torch.logical_and(img_knn_overlap_masks_2d, img_knn_overlap_masks_3d)
    img_knn_overlap_masks = torch.logical_and(img_knn_overlap_masks, img_knn_min_pcd_masks)
    img_knn_overlap_masks = torch.logical_and(img_knn_overlap_masks, img_knn_masks)

    # compute 3d overlap masks
    pcd_knn_min_distances_3d, pcd_knn_min_indices_3d = knn(pcd_knn_points, img_knn_points, k=1, return_distance=True)
    pcd_knn_min_distances_3d = pcd_knn_min_distances_3d.squeeze(-1)
    pcd_knn_min_indices_3d = pcd_knn_min_indices_3d.squeeze(-1)
    pcd_knn_batch_indices_3d = torch.arange(num_candidates).cuda().unsqueeze(1).expand_as(pcd_knn_min_indices_3d)
    pcd_knn_min_img_pixels = img_knn_pixels[pcd_knn_batch_indices_3d, pcd_knn_min_indices_3d]
    pcd_knn_min_distances_2d = torch.linalg.norm(pcd_knn_pixels - pcd_knn_min_img_pixels, dim=-1)
    pcd_knn_min_img_masks = img_knn_masks[pcd_knn_batch_indices_3d, pcd_knn_min_indices_3d]
    pcd_knn_overlap_masks_3d = torch.lt(pcd_knn_min_distances_3d, pos_radius_3d)
    pcd_knn_overlap_masks_2d = torch.lt(pcd_knn_min_distances_2d, pos_radius_2d)
    pcd_knn_overlap_masks = torch.logical_and(pcd_knn_overlap_masks_2d, pcd_knn_overlap_masks_3d)
    pcd_knn_overlap_masks = torch.logical_and(pcd_knn_overlap_masks, pcd_knn_min_img_masks)
    pcd_knn_overlap_masks = torch.logical_and(pcd_knn_overlap_masks, pcd_knn_masks)

    # compute overlaps
    img_overlap_counts = img_knn_overlap_masks.sum(1)  # (B,)
    pcd_overlap_counts = pcd_knn_overlap_masks.sum(1)  # (B,)
    img_total_counts = img_knn_masks.sum(-1)  # (B,)
    pcd_total_counts = pcd_knn_masks.sum(-1)  # (B,)
    img_overlap_ratios = img_overlap_counts.float() / img_total_counts.float()  # (B,)
    pcd_overlap_ratios = pcd_overlap_counts.float() / pcd_total_counts.float()  # (B,)

    img_overlap_masks = torch.gt(img_overlap_ratios, 0.0)
    pcd_overlap_masks = torch.gt(pcd_overlap_ratios, 0.0)
    overlap_masks = torch.logical_and(img_overlap_masks, pcd_overlap_masks)
    img_corr_indices = candidate_img_indices[overlap_masks]
    pcd_corr_indices = candidate_pcd_indices[overlap_masks]
    img_corr_overlaps = img_overlap_ratios[overlap_masks]
    pcd_corr_overlaps = pcd_overlap_ratios[overlap_masks]

    return img_corr_indices, pcd_corr_indices, img_corr_overlaps, pcd_corr_overlaps
