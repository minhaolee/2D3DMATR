import glob
import json
import os.path as osp
import time

import numpy as np

# isort: split
from config import make_cfg

from vision3d.array_ops import (
    evaluate_correspondences,
    evaluate_sparse_correspondences,
    isotropic_registration_error,
    registration_rmse,
)
from vision3d.utils.logger import get_logger
from vision3d.utils.opencv import registration_with_pnp_ransac
from vision3d.utils.parser import get_default_parser
from vision3d.utils.summary_board import SummaryBoard


def make_parser():
    parser = get_default_parser()
    parser.add_argument("--test_epoch", type=int, required=True, help="test epoch")
    parser.add_argument("--num_corr", type=int, default=None, help="number of correspondences for registration")
    parser.add_argument("--verbose", action="store_true", help="verbose mode")
    return parser


def eval_one_epoch(args, cfg, logger):
    cache_dir = cfg.exp.cache_dir

    coarse_matching_meter = SummaryBoard()
    coarse_matching_meter.register_meter("PIR")
    coarse_matching_meter.register_meter("PMR>0")
    coarse_matching_meter.register_meter("PMR>=0.1")
    coarse_matching_meter.register_meter("PMR>=0.3")
    coarse_matching_meter.register_meter("PMR>=0.5")
    coarse_matching_meter.register_meter("scene_PIR")
    coarse_matching_meter.register_meter("scene_PMR>0")
    coarse_matching_meter.register_meter("scene_PMR>=0.1")
    coarse_matching_meter.register_meter("scene_PMR>=0.3")
    coarse_matching_meter.register_meter("scene_PMR>=0.5")

    fine_matching_meter = SummaryBoard()
    fine_matching_meter.register_meter("FMR")
    fine_matching_meter.register_meter("IR")
    fine_matching_meter.register_meter("OR")
    fine_matching_meter.register_meter("scene_FMR")
    fine_matching_meter.register_meter("scene_IR")
    fine_matching_meter.register_meter("scene_OR")

    registration_meter = SummaryBoard()
    registration_meter.register_meter("RR")
    registration_meter.register_meter("mean_RRE")
    registration_meter.register_meter("mean_RTE")
    registration_meter.register_meter("median_RRE")
    registration_meter.register_meter("median_RTE")
    registration_meter.register_meter("scene_RR")
    registration_meter.register_meter("scene_RRE")
    registration_meter.register_meter("scene_RTE")

    scene_coarse_matching_result_dict = {}
    scene_fine_matching_result_dict = {}
    scene_registration_result_dict = {}

    scene_dirs = sorted(glob.glob(osp.join(cache_dir, "*")))
    for scene_dir in scene_dirs:
        coarse_matching_meter.reset_meter("scene_PIR")
        coarse_matching_meter.reset_meter("scene_PMR>0")
        coarse_matching_meter.reset_meter("scene_PMR>=0.1")
        coarse_matching_meter.reset_meter("scene_PMR>=0.3")
        coarse_matching_meter.reset_meter("scene_PMR>=0.5")

        fine_matching_meter.reset_meter("scene_FMR")
        fine_matching_meter.reset_meter("scene_IR")
        fine_matching_meter.reset_meter("scene_overlap")

        registration_meter.reset_meter("scene_RR")
        registration_meter.reset_meter("scene_RRE")
        registration_meter.reset_meter("scene_RTE")

        scene_name = osp.basename(scene_dir)

        file_names = sorted(glob.glob(osp.join(scene_dir, "*.npz")))
        for file_name in file_names:
            split_name = [x for x in osp.basename(file_name).split(".")[0].split("_")]
            image_name = split_name[0]
            cloud_name = split_name[1]

            data_dict = np.load(file_name)

            pcd_points = data_dict["pcd_points"]

            img_num_nodes = data_dict["img_num_nodes"]
            pcd_num_nodes = data_dict["pcd_num_nodes"]
            img_node_corr_indices = data_dict["img_node_corr_indices"]
            pcd_node_corr_indices = data_dict["pcd_node_corr_indices"]

            img_corr_points = data_dict["img_corr_points"]
            pcd_corr_points = data_dict["pcd_corr_points"]
            img_corr_pixels = data_dict["img_corr_pixels"]
            pcd_corr_pixels = data_dict["pcd_corr_pixels"]
            corr_scores = data_dict["corr_scores"]

            gt_img_node_corr_indices = data_dict["gt_img_node_corr_indices"]
            gt_pcd_node_corr_indices = data_dict["gt_pcd_node_corr_indices"]
            transform = data_dict["transform"]

            # aligned_pcd_corr_points = apply_transform(pcd_corr_points, transform)
            # pos_corr_masks = np.linalg.norm(aligned_pcd_corr_points - img_corr_points, axis=1) < 0.05
            # pos_img_corr_points = img_corr_points[pos_corr_masks]
            # pos_img_corr_pixels = img_corr_pixels[pos_corr_masks]
            # pos_pcd_corr_points = aligned_pcd_corr_points[pos_corr_masks]
            # pos_pcd_corr_pixels = pcd_corr_pixels[pos_corr_masks]
            # pos_corr_scores = corr_scores[pos_corr_masks]
            # pos_residual_2d = np.linalg.norm(pos_pcd_corr_pixels - pos_img_corr_pixels, axis=1).mean()
            # pos_residual_3d = np.linalg.norm(pos_pcd_corr_points - pos_img_corr_points, axis=1).mean()
            # pos_num_correspondences = pos_corr_masks.sum()
            # print(f"RS_2d: {pos_residual_2d:.3f}, RS_3d: {pos_residual_3d}, NC: {pos_num_correspondences}")

            if args.num_corr is not None and corr_scores.shape[0] > args.num_corr:
                num_corr = corr_scores.shape[0]
                sel_indices = np.argsort(-corr_scores)[:num_corr]
                img_corr_points = img_corr_points[sel_indices]
                pcd_corr_points = pcd_corr_points[sel_indices]
                img_corr_pixels = img_corr_pixels[sel_indices]
                pcd_corr_pixels = pcd_corr_pixels[sel_indices]
                corr_scores = corr_scores[sel_indices]

            num_correspondences = img_corr_points.shape[0]

            message = f"{scene_name}, img: {image_name}, pcd: {cloud_name}"

            # 1. evaluate correspondences
            # 1.1 evaluate coarse correspondences
            coarse_matching_result_dict = evaluate_sparse_correspondences(
                img_num_nodes,
                pcd_num_nodes,
                img_node_corr_indices,
                pcd_node_corr_indices,
                gt_img_node_corr_indices,
                gt_pcd_node_corr_indices,
            )

            coarse_precision = coarse_matching_result_dict["precision"]

            coarse_matching_meter.update("scene_PIR", coarse_precision)
            coarse_matching_meter.update("scene_PMR>0", float(coarse_precision > 0))
            coarse_matching_meter.update("scene_PMR>=0.1", float(coarse_precision >= 0.1))
            coarse_matching_meter.update("scene_PMR>=0.3", float(coarse_precision >= 0.3))
            coarse_matching_meter.update("scene_PMR>=0.5", float(coarse_precision >= 0.5))

            # 1.2 evaluate fine correspondences
            if num_correspondences > 0:
                fine_matching_result_dict = evaluate_correspondences(
                    pcd_corr_points, img_corr_points, transform, positive_radius=cfg.eval.acceptance_radius
                )
            else:
                fine_matching_result_dict = {"inlier_ratio": 0.0, "overlap": 0.0, "distance": 0.0}

            inlier_ratio = fine_matching_result_dict["inlier_ratio"]
            overlap = fine_matching_result_dict["overlap"]

            fine_matching_meter.update("scene_IR", inlier_ratio)
            fine_matching_meter.update("scene_OR", overlap)
            fine_matching_meter.update("scene_FMR", float(inlier_ratio >= cfg.eval.inlier_ratio_threshold))

            message += ", c_PIR: {:.3f}".format(coarse_precision)
            message += ", f_IR: {:.3f}".format(inlier_ratio)
            message += ", f_OR: {:.3f}".format(overlap)
            message += ", f_RS: {:.3f}".format(fine_matching_result_dict["distance"])
            message += ", f_NU: {}".format(num_correspondences)

            # 2. evaluate registration
            if num_correspondences >= 4:
                intrinsics = data_dict["intrinsics"]
                estimated_transform = registration_with_pnp_ransac(
                    pcd_corr_points,
                    img_corr_pixels,
                    intrinsics,
                    num_iterations=cfg.ransac.num_iterations,
                    distance_tolerance=cfg.ransac.distance_tolerance,
                )

                rmse = registration_rmse(pcd_points, transform, estimated_transform)
                registration_recall = float(rmse < cfg.eval.rmse_threshold)
                message += ", r_RMSE: {:.3f}".format(rmse)
            else:
                estimated_transform = None
                registration_recall = 0.0

            registration_meter.update("scene_RR", registration_recall)
            message += ", r_RR: {:.3f}".format(registration_recall)

            if registration_recall > 0.0:
                rre, rte = isotropic_registration_error(transform, estimated_transform)
                registration_meter.update("scene_RRE", rre)
                registration_meter.update("scene_RTE", rte)
                message += ", r_RRE: {:.3f}".format(rre)
                message += ", r_RTE: {:.6f}".format(rte)

            if args.verbose:
                logger.info(message)

        logger.info(f"Scene_name: {scene_name}")

        # 1. print correspondence evaluation results (one scene)
        # 1.1 coarse level statistics
        coarse_precision = coarse_matching_meter.mean("scene_PIR")
        coarse_matching_recall_0 = coarse_matching_meter.mean("scene_PMR>0")
        coarse_matching_recall_1 = coarse_matching_meter.mean("scene_PMR>=0.1")
        coarse_matching_recall_3 = coarse_matching_meter.mean("scene_PMR>=0.3")
        coarse_matching_recall_5 = coarse_matching_meter.mean("scene_PMR>=0.5")
        coarse_matching_meter.update("PIR", coarse_precision)
        coarse_matching_meter.update("PMR>0", coarse_matching_recall_0)
        coarse_matching_meter.update("PMR>=0.1", coarse_matching_recall_1)
        coarse_matching_meter.update("PMR>=0.3", coarse_matching_recall_3)
        coarse_matching_meter.update("PMR>=0.5", coarse_matching_recall_5)
        scene_coarse_matching_result_dict[scene_name] = {
            "PIR": coarse_precision,
            "PMR>0": coarse_matching_recall_0,
            "PMR>=0.1": coarse_matching_recall_1,
            "PMR>=0.3": coarse_matching_recall_3,
            "PMR>=0.5": coarse_matching_recall_5,
        }

        # 1.2 fine level statistics
        recall = fine_matching_meter.mean("scene_FMR")
        inlier_ratio = fine_matching_meter.mean("scene_IR")
        overlap = fine_matching_meter.mean("scene_OR")
        fine_matching_meter.update("FMR", recall)
        fine_matching_meter.update("IR", inlier_ratio)
        fine_matching_meter.update("OR", overlap)
        scene_fine_matching_result_dict[scene_name] = {"FMR": recall, "IR": inlier_ratio}

        message = "  Correspondence"
        message += ", c_PIR: {:.3f}".format(coarse_precision)
        message += ", c_PMR>0: {:.3f}".format(coarse_matching_recall_0)
        message += ", c_PMR>=0.1: {:.3f}".format(coarse_matching_recall_1)
        message += ", c_PMR>=0.3: {:.3f}".format(coarse_matching_recall_3)
        message += ", c_PMR>=0.5: {:.3f}".format(coarse_matching_recall_5)
        message += ", f_FMR: {:.3f}".format(recall)
        message += ", f_IR: {:.3f}".format(inlier_ratio)
        message += ", f_OR: {:.3f}".format(overlap)
        logger.info(message)

        # 2. print registration evaluation results (one scene)
        recall = registration_meter.mean("scene_RR")
        mean_rre = registration_meter.mean("scene_RRE")
        mean_rte = registration_meter.mean("scene_RTE")
        median_rre = registration_meter.median("scene_RRE")
        median_rte = registration_meter.median("scene_RTE")
        registration_meter.update("RR", recall)
        registration_meter.update("mean_RRE", mean_rre)
        registration_meter.update("mean_RTE", mean_rte)
        registration_meter.update("median_RRE", median_rre)
        registration_meter.update("median_RTE", median_rte)

        scene_registration_result_dict[scene_name] = {
            "RR": recall,
            "mean_RRE": mean_rre,
            "mean_RTE": mean_rte,
            "median_RRE": median_rre,
            "median_RTE": median_rte,
        }

        message = "  Registration"
        message += ", RR: {:.3f}".format(recall)
        message += ", mean_RRE: {:.3f}".format(mean_rre)
        message += ", mean_RTE: {:.3f}".format(mean_rte)
        message += ", median_RRE: {:.3f}".format(median_rre)
        message += ", median_RTE: {:.3f}".format(median_rte)
        logger.info(message)

    logger.success("Epoch {}".format(args.test_epoch))

    # 1. print correspondence evaluation results
    message = "  Coarse Matching"
    message += ", PIR: {:.3f}".format(coarse_matching_meter.mean("PIR"))
    message += ", PMR>0: {:.3f}".format(coarse_matching_meter.mean("PMR>0"))
    message += ", PMR>=0.1: {:.3f}".format(coarse_matching_meter.mean("PMR>=0.1"))
    message += ", PMR>=0.3: {:.3f}".format(coarse_matching_meter.mean("PMR>=0.3"))
    message += ", PMR>=0.5: {:.3f}".format(coarse_matching_meter.mean("PMR>=0.5"))
    logger.success(message)
    for scene_name, result_dict in scene_coarse_matching_result_dict.items():
        message = "    {}".format(scene_name)
        message += ", PIR: {:.3f}".format(result_dict["PIR"])
        message += ", PMR>0: {:.3f}".format(result_dict["PMR>0"])
        message += ", PMR>=0.1: {:.3f}".format(result_dict["PMR>=0.1"])
        message += ", PMR>=0.3: {:.3f}".format(result_dict["PMR>=0.3"])
        message += ", PMR>=0.5: {:.3f}".format(result_dict["PMR>=0.5"])
        logger.success(message)

    message = "  Fine Matching"
    message += ", FMR: {:.3f}".format(fine_matching_meter.mean("FMR"))
    message += ", IR: {:.3f}".format(fine_matching_meter.mean("IR"))
    message += ", OR: {:.3f}".format(fine_matching_meter.mean("OR"))
    message += ", std: {:.3f}".format(fine_matching_meter.std("FMR"))
    logger.success(message)
    for scene_name, result_dict in scene_fine_matching_result_dict.items():
        message = "    {}".format(scene_name)
        message += ", FMR: {:.3f}".format(result_dict["FMR"])
        message += ", IR: {:.3f}".format(result_dict["IR"])
        logger.success(message)

    # 2. print registration evaluation results
    message = "  Registration"
    message += ", RR: {:.3f}".format(registration_meter.mean("RR"))
    message += ", mean_RRE: {:.3f}".format(registration_meter.mean("mean_RRE"))
    message += ", mean_RTE: {:.3f}".format(registration_meter.mean("mean_RTE"))
    message += ", median_RRE: {:.3f}".format(registration_meter.mean("median_RRE"))
    message += ", median_RTE: {:.3f}".format(registration_meter.mean("median_RTE"))
    logger.success(message)
    for scene_name, result_dict in scene_registration_result_dict.items():
        message = "    {}".format(scene_name)
        message += ", RR: {:.3f}".format(result_dict["RR"])
        message += ", mean_RRE: {:.3f}".format(result_dict["mean_RRE"])
        message += ", mean_RTE: {:.3f}".format(result_dict["mean_RTE"])
        message += ", median_RRE: {:.3f}".format(result_dict["median_RRE"])
        message += ", median_RTE: {:.3f}".format(result_dict["median_RTE"])
        logger.success(message)

    summary_dict = {
        "log_file": osp.basename(logger.log_file),
        "epoch": args.test_epoch,
        "PIR": coarse_matching_meter.mean("PIR"),
        "FMR": fine_matching_meter.mean("FMR"),
        "IR": fine_matching_meter.mean("IR"),
        "RR": registration_meter.mean("RR"),
        "mean_RRE": registration_meter.mean("mean_RRE"),
        "mean_RTE": registration_meter.mean("mean_RTE"),
        "median_RRE": registration_meter.mean("median_RRE"),
        "median_RTE": registration_meter.mean("median_RTE"),
    }
    with open(osp.join(cfg.exp.log_dir, "summary.json"), "a") as f:
        result_item = json.dumps(summary_dict) + "\n"
        f.write(result_item)


def main():
    parser = make_parser()
    args = parser.parse_args()

    cfg = make_cfg()
    log_file = osp.join(cfg.exp.log_dir, "eval-{}.log".format(time.strftime("%Y%m%d-%H%M%S")))
    logger = get_logger(log_file=log_file)

    message = "Configs:\n" + json.dumps(cfg, indent=4)
    logger.info(message)

    eval_one_epoch(args, cfg, logger)


if __name__ == "__main__":
    main()
