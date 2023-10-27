from typing import Union

import torch.nn as nn
import torch.nn.functional as F

from vision3d.layers import ConvBlock, build_act_layer


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dilation: int = 1,
        norm_cfg: Union[str, dict] = "GroupNorm",
        act_cfg: Union[str, dict] = "LeakyReLU",
    ):
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=stride,
            dilation=dilation,
            conv_cfg="Conv2d",
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.conv2 = ConvBlock(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            dilation=dilation,
            conv_cfg="Conv2d",
            norm_cfg=norm_cfg,
            act_cfg="None",
        )

        if stride == 1:
            self.identity = nn.Identity()
        else:
            self.identity = ConvBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
                dilation=dilation,
                conv_cfg="Conv2d",
                norm_cfg=norm_cfg,
                act_cfg="None",
            )

        self.act = build_act_layer(act_cfg)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        identity = self.identity(x)
        output = self.act(identity + residual)
        return output


class ImageBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int,
        dilation: int = 1,
        norm_cfg: Union[str, dict] = "GroupNorm",
        act_cfg: Union[str, dict] = "LeakyReLU",
    ):
        super().__init__()

        self.encoder1 = ConvBlock(
            in_channels,
            base_channels * 1,
            kernel_size=7,
            padding=3,
            stride=2,
            conv_cfg="Conv2d",
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.encoder2 = nn.Sequential(
            BasicBlock(
                base_channels * 1,
                base_channels * 1,
                stride=1,
                dilation=dilation,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            BasicBlock(
                base_channels * 1,
                base_channels * 1,
                stride=1,
                dilation=dilation,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )

        self.encoder3 = nn.Sequential(
            BasicBlock(
                base_channels * 1,
                base_channels * 2,
                stride=2,
                dilation=dilation,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            BasicBlock(
                base_channels * 2,
                base_channels * 2,
                stride=1,
                dilation=dilation,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )

        self.encoder4 = nn.Sequential(
            BasicBlock(
                base_channels * 2,
                base_channels * 4,
                stride=2,
                dilation=dilation,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            BasicBlock(
                base_channels * 4,
                base_channels * 4,
                stride=1,
                dilation=dilation,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
        )

        self.decoder4_1 = ConvBlock(
            base_channels * 4,
            base_channels * 4,
            kernel_size=1,
            conv_cfg="Conv2d",
            norm_cfg="None",
            act_cfg="None",
        )

        self.decoder3_1 = ConvBlock(
            base_channels * 2,
            base_channels * 4,
            kernel_size=1,
            conv_cfg="Conv2d",
            norm_cfg="None",
            act_cfg="None",
        )
        self.decoder3_2 = nn.Sequential(
            ConvBlock(
                base_channels * 4,
                base_channels * 4,
                kernel_size=3,
                padding=1,
                conv_cfg="Conv2d",
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvBlock(
                base_channels * 4,
                base_channels * 2,
                kernel_size=3,
                padding=1,
                conv_cfg="Conv2d",
                norm_cfg="None",
                act_cfg="None",
            ),
        )

        self.decoder2_1 = ConvBlock(
            base_channels * 1,
            base_channels * 2,
            kernel_size=1,
            conv_cfg="Conv2d",
            norm_cfg="None",
            act_cfg="None",
        )
        self.decoder2_2 = nn.Sequential(
            ConvBlock(
                base_channels * 2,
                base_channels * 2,
                kernel_size=3,
                padding=1,
                conv_cfg="Conv2d",
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvBlock(
                base_channels * 2,
                base_channels * 1,
                kernel_size=3,
                padding=1,
                conv_cfg="Conv2d",
                norm_cfg="None",
                act_cfg="None",
            ),
        )

        self.decoder1_1 = ConvBlock(
            base_channels * 1,
            base_channels * 1,
            kernel_size=1,
            conv_cfg="Conv2d",
            norm_cfg="None",
            act_cfg="None",
        )
        self.decoder1_2 = nn.Sequential(
            ConvBlock(
                base_channels * 1,
                base_channels * 1,
                kernel_size=3,
                padding=1,
                conv_cfg="Conv2d",
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvBlock(
                base_channels * 1,
                base_channels * 1,
                kernel_size=3,
                padding=1,
                conv_cfg="Conv2d",
                norm_cfg="None",
                act_cfg="None",
            ),
        )

        self.out_proj = ConvBlock(
            base_channels * 1,
            out_channels,
            kernel_size=1,
            conv_cfg="Conv2d",
            norm_cfg="None",
            act_cfg="None",
        )

    def forward(self, x):
        feats_list = []

        # encoder
        feats_s1 = self.encoder1(x)  # 1/2
        feats_s2 = self.encoder2(feats_s1)  # 1/2
        feats_s3 = self.encoder3(feats_s2)  # 1/4
        feats_s4 = self.encoder4(feats_s3)  # 1/8

        # decoder
        latent_s4 = self.decoder4_1(feats_s4)  # (1/8)
        feats_list.append(latent_s4)

        interp_s3 = F.interpolate(latent_s4, size=feats_s3.shape[2:], mode="bilinear", align_corners=True)
        latent_s3 = self.decoder3_1(feats_s3)
        latent_s3 = self.decoder3_2(latent_s3 + interp_s3)
        feats_list.append(latent_s3)

        interp_s2 = F.interpolate(latent_s3, size=feats_s2.shape[2:], mode="bilinear", align_corners=True)
        latent_s2 = self.decoder2_1(feats_s2)
        latent_s2 = self.decoder2_2(latent_s2 + interp_s2)
        feats_list.append(latent_s2)

        latent_s1 = self.decoder1_1(feats_s1) + latent_s2
        interp_s1 = F.interpolate(latent_s1, size=x.shape[2:], mode="bilinear", align_corners=True)
        latent_s1 = self.decoder1_2(interp_s1)

        latent_s1 = self.out_proj(latent_s1)
        feats_list.append(latent_s1)

        feats_list.reverse()

        return feats_list


class FeaturePyramid(nn.Module):
    def __init__(self, d_model, norm_cfg="GroupNorm", act_cfg="LeakyReLU"):
        super().__init__()

        self.latent_1 = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.latent_2 = nn.Conv2d(d_model, d_model, kernel_size=1)
        self.latent_3 = nn.Conv2d(d_model, d_model, kernel_size=1)

        self.downsample1 = BasicBlock(d_model, d_model, stride=2, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.downsample2 = BasicBlock(d_model, d_model, stride=2, norm_cfg=norm_cfg, act_cfg=act_cfg)

    def forward(self, feats):
        feats_s1 = feats
        feats_s2 = self.downsample1(feats_s1)
        feats_s3 = self.downsample2(feats_s2)

        feats_s1 = self.latent_1(feats_s1)
        feats_s2 = self.latent_2(feats_s2)
        feats_s3 = self.latent_3(feats_s3)

        return feats_s1, feats_s2, feats_s3


def run_test():
    model = ImageBackbone(1, 128, 128)
    print(model)


if __name__ == "__main__":
    run_test()
