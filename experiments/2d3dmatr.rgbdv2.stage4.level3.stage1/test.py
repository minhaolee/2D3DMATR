import os.path as osp
import time

import numpy as np
from config import make_cfg
from dataset import test_data_loader
from loss import EvalFunction
from model import create_model

from vision3d.engine import SingleTester
from vision3d.utils.io import ensure_dir
from vision3d.utils.misc import get_log_string
from vision3d.utils.parser import add_tester_args
from vision3d.utils.tensor import tensor_to_array


class Tester(SingleTester):
    def __init__(self, cfg):
        super().__init__(cfg)

        # dataloader
        start_time = time.time()
        data_loader, neighbor_limits = test_data_loader(cfg)
        loading_time = time.time() - start_time
        self.log(f"Data loader created: {loading_time:.3f}s collapsed.", level="DEBUG")
        self.log(f"Calibrate neighbors: {neighbor_limits}.")
        self.register_loader(data_loader)

        # model
        model = create_model(cfg).cuda()
        self.register_model(model)

        # evaluator
        self.eval_func = EvalFunction(cfg).cuda()

        # preparation
        self.output_dir = cfg.exp.cache_dir

    def test_step(self, iteration, data_dict):
        output_dict = self.model(data_dict)
        return output_dict

    def eval_step(self, iteration, data_dict, output_dict):
        result_dict = self.eval_func(data_dict, output_dict)
        result_dict["duration"] = output_dict["duration"]
        return result_dict

    def get_log_string(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict["scene_name"]
        image_id = data_dict["image_id"]
        cloud_id = data_dict["cloud_id"]
        message = f"{scene_name}, img: {image_id}, pcd: {cloud_id}"
        message += ", " + get_log_string(result_dict=result_dict)
        message += ", nCorr: {}".format(output_dict["corr_scores"].shape[0])
        return message

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        scene_name = data_dict["scene_name"]
        image_id = data_dict["image_id"]
        cloud_id = data_dict["cloud_id"]

        ensure_dir(osp.join(self.output_dir, scene_name))
        file_name = osp.join(self.output_dir, scene_name, f"{image_id}_{cloud_id}.npz")
        np.savez_compressed(
            file_name,
            image_file=data_dict["image_file"],
            depth_file=data_dict["depth_file"],
            cloud_file=data_dict["cloud_file"],
            pcd_points=tensor_to_array(output_dict["pcd_points"]),
            pcd_points_f=tensor_to_array(output_dict["pcd_points_f"]),
            pcd_points_c=tensor_to_array(output_dict["pcd_points_c"]),
            img_num_nodes=output_dict["img_num_nodes"],
            pcd_num_nodes=output_dict["pcd_num_nodes"],
            img_node_corr_indices=tensor_to_array(output_dict["img_node_corr_indices"]),
            pcd_node_corr_indices=tensor_to_array(output_dict["pcd_node_corr_indices"]),
            img_node_corr_levels=tensor_to_array(output_dict["img_node_corr_levels"]),
            img_corr_points=tensor_to_array(output_dict["img_corr_points"]),
            pcd_corr_points=tensor_to_array(output_dict["pcd_corr_points"]),
            img_corr_pixels=tensor_to_array(output_dict["img_corr_pixels"]),
            pcd_corr_pixels=tensor_to_array(output_dict["pcd_corr_pixels"]),
            corr_scores=tensor_to_array(output_dict["corr_scores"]),
            gt_img_node_corr_indices=tensor_to_array(output_dict["gt_img_node_corr_indices"]),
            gt_pcd_node_corr_indices=tensor_to_array(output_dict["gt_pcd_node_corr_indices"]),
            gt_img_node_corr_overlaps=tensor_to_array(output_dict["gt_img_node_corr_overlaps"]),
            gt_pcd_node_corr_overlaps=tensor_to_array(output_dict["gt_pcd_node_corr_overlaps"]),
            gt_node_corr_min_overlaps=tensor_to_array(output_dict["gt_node_corr_min_overlaps"]),
            gt_node_corr_max_overlaps=tensor_to_array(output_dict["gt_node_corr_max_overlaps"]),
            transform=tensor_to_array(data_dict["transform"]),
            intrinsics=tensor_to_array(data_dict["intrinsics"]),
            overlap=data_dict["overlap"],
        )


def main():
    add_tester_args()
    cfg = make_cfg()
    tester = Tester(cfg)
    tester.run()


if __name__ == "__main__":
    main()
