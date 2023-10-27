import torch
import torch.nn as nn

from vision3d.layers import KPConvBlock, KPResidualBlock, UnaryBlockPackMode
from vision3d.ops import knn_interpolate_pack_mode


class PointBackbone(nn.Module):
    def __init__(self, input_dim, output_dim, init_dim, kernel_size, init_radius, init_sigma):
        super().__init__()

        self.encoder1_1 = KPConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma)
        self.encoder1_2 = KPResidualBlock(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma)

        self.encoder2_1 = KPResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, strided=True
        )
        self.encoder2_2 = KPResidualBlock(init_dim * 2, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2)
        self.encoder2_3 = KPResidualBlock(init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2)

        self.encoder3_1 = KPResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, strided=True
        )
        self.encoder3_2 = KPResidualBlock(init_dim * 4, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4)
        self.encoder3_3 = KPResidualBlock(init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4)

        self.encoder4_1 = KPResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, strided=True
        )
        self.encoder4_2 = KPResidualBlock(init_dim * 8, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8)
        self.encoder4_3 = KPResidualBlock(init_dim * 16, init_dim * 16, kernel_size, init_radius * 8, init_sigma * 8)

        self.decoder3 = UnaryBlockPackMode(init_dim * 24, init_dim * 8)
        self.decoder2 = UnaryBlockPackMode(init_dim * 12, init_dim * 4)
        self.decoder1 = UnaryBlockPackMode(init_dim * 6, init_dim * 2)

        self.out_proj = nn.Linear(init_dim * 2, output_dim)

    def forward(self, feats, data_dict):
        feats_list = []

        points_list = data_dict["points"]
        neighbors_list = data_dict["neighbors"]
        subsampling_list = data_dict["subsampling"]
        upsampling_list = data_dict["upsampling"]

        feats_s1 = feats
        feats_s1 = self.encoder1_1(points_list[0], points_list[0], feats_s1, neighbors_list[0])
        feats_s1 = self.encoder1_2(points_list[0], points_list[0], feats_s1, neighbors_list[0])

        feats_s2 = self.encoder2_1(points_list[1], points_list[0], feats_s1, subsampling_list[0])
        feats_s2 = self.encoder2_2(points_list[1], points_list[1], feats_s2, neighbors_list[1])
        feats_s2 = self.encoder2_3(points_list[1], points_list[1], feats_s2, neighbors_list[1])

        feats_s3 = self.encoder3_1(points_list[2], points_list[1], feats_s2, subsampling_list[1])
        feats_s3 = self.encoder3_2(points_list[2], points_list[2], feats_s3, neighbors_list[2])
        feats_s3 = self.encoder3_3(points_list[2], points_list[2], feats_s3, neighbors_list[2])

        feats_s4 = self.encoder4_1(points_list[3], points_list[2], feats_s3, subsampling_list[2])
        feats_s4 = self.encoder4_2(points_list[3], points_list[3], feats_s4, neighbors_list[3])
        feats_s4 = self.encoder4_3(points_list[3], points_list[3], feats_s4, neighbors_list[3])

        latent_s4 = feats_s4
        feats_list.append(latent_s4)

        latent_s3 = knn_interpolate_pack_mode(points_list[2], points_list[3], latent_s4, upsampling_list[2])
        latent_s3 = torch.cat([latent_s3, feats_s3], dim=1)
        latent_s3 = self.decoder3(latent_s3)
        feats_list.append(latent_s3)

        latent_s2 = knn_interpolate_pack_mode(points_list[1], points_list[2], latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)
        feats_list.append(latent_s2)

        latent_s1 = knn_interpolate_pack_mode(points_list[0], points_list[1], latent_s2, upsampling_list[0])
        latent_s1 = torch.cat([latent_s1, feats_s1], dim=1)
        latent_s1 = self.decoder1(latent_s1)

        latent_s1 = self.out_proj(latent_s1)
        feats_list.append(latent_s1)

        feats_list.reverse()

        return feats_list
