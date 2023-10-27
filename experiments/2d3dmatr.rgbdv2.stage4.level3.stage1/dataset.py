from vision3d.datasets.registration import SevenScenes2D3DHardPairDataset
from vision3d.utils.collate import GraphPyramid2D3DRegistrationCollateFn
from vision3d.utils.dataloader import build_dataloader, calibrate_neighbors_pack_mode


def train_valid_data_loader(cfg):
    train_dataset = SevenScenes2D3DHardPairDataset(
        cfg.data.dataset_dir,
        "train",
        max_points=cfg.train.max_points,
        return_corr_indices=True,
        matching_radius_2d=cfg.data.matching_radius_2d,
        matching_radius_3d=cfg.data.matching_radius_3d,
        scene_name=cfg.train.scene_name,
        overlap_threshold=cfg.data.overlap_threshold,
        use_augmentation=True,
    )

    neighbor_limits = calibrate_neighbors_pack_mode(
        train_dataset,
        GraphPyramid2D3DRegistrationCollateFn,
        cfg.model.point_backbone.num_stages,
        cfg.model.point_backbone.base_voxel_size,
        cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
    )

    collate_fn = GraphPyramid2D3DRegistrationCollateFn(
        cfg.model.point_backbone.num_stages,
        cfg.model.point_backbone.base_voxel_size,
        cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
        neighbor_limits,
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )

    valid_dataset = SevenScenes2D3DHardPairDataset(
        cfg.data.dataset_dir,
        "val",
        max_points=cfg.test.max_points,
        return_corr_indices=True,
        matching_radius_2d=cfg.data.matching_radius_2d,
        matching_radius_3d=cfg.data.matching_radius_3d,
        scene_name=cfg.test.scene_name,
        overlap_threshold=cfg.data.overlap_threshold,
    )

    valid_loader = build_dataloader(
        valid_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg):
    train_dataset = SevenScenes2D3DHardPairDataset(
        cfg.data.dataset_dir,
        "train",
        max_points=cfg.train.max_points,
        return_corr_indices=True,
        matching_radius_2d=cfg.data.matching_radius_2d,
        matching_radius_3d=cfg.data.matching_radius_3d,
        scene_name=cfg.train.scene_name,
        overlap_threshold=cfg.data.overlap_threshold,
        use_augmentation=True,
    )

    neighbor_limits = calibrate_neighbors_pack_mode(
        train_dataset,
        GraphPyramid2D3DRegistrationCollateFn,
        cfg.model.point_backbone.num_stages,
        cfg.model.point_backbone.base_voxel_size,
        cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
    )

    collate_fn = GraphPyramid2D3DRegistrationCollateFn(
        cfg.model.point_backbone.num_stages,
        cfg.model.point_backbone.base_voxel_size,
        cfg.model.point_backbone.base_voxel_size * cfg.model.point_backbone.kpconv_radius,
        neighbor_limits,
    )

    test_dataset = SevenScenes2D3DHardPairDataset(
        cfg.data.dataset_dir,
        "test",
        max_points=cfg.test.max_points,
        return_corr_indices=True,
        matching_radius_2d=cfg.data.matching_radius_2d,
        matching_radius_3d=cfg.data.matching_radius_3d,
        scene_name=cfg.test.scene_name,
        overlap_threshold=cfg.data.overlap_threshold,
    )

    test_loader = build_dataloader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    return test_loader, neighbor_limits


def run_test():
    import numpy as np

    from vision3d.array_ops import apply_transform, back_project
    from vision3d.utils.open3d import draw_geometries, get_color, make_open3d_point_cloud
    from vision3d.utils.tensor import tensor_to_array
    from vision3d.utils.visualization import draw_correspondences

    # isort: split
    from config import make_cfg

    cfg = make_cfg()
    train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg)
    print(neighbor_limits)
    for data_dict in train_loader:
        data_dict = tensor_to_array(data_dict)

        depth = data_dict["depth"][0]
        intrinsic = data_dict["intrinsic"]
        img_points, img_pixels = back_project(depth, intrinsic, return_pixels=True)

        pcd_points = data_dict["points"][0]
        transform = data_dict["transform"]
        pcd_points = apply_transform(pcd_points, transform)

        img_corr_pixels = data_dict["img_corr_pixels"]
        pcd_corr_indices = data_dict["pcd_corr_indices"]

        img_indices = np.full_like(depth, fill_value=-1, dtype=np.int)
        img_indices[img_pixels[:, 0], img_pixels[:, 1]] = np.arange(img_pixels.shape[0])
        img_corr_indices = img_indices[img_corr_pixels[:, 0], img_corr_pixels[:, 1]]

        # img_corr_points = img_points[img_corr_indices]
        # pcd_corr_points = pcd_points[pcd_corr_indices]

        draw_correspondences(img_points, pcd_points, img_corr_indices, pcd_corr_indices)


if __name__ == "__main__":
    run_test()
