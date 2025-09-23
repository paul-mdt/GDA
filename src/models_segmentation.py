from __future__ import annotations

"""Segmentation model definitions."""

import math
from typing import Iterable, Optional, Sequence, Tuple, Union

import einops
import torch
import torch.nn.functional as F


def _build_norm_layer(num_features: int, norm_cfg: Optional[dict]):
    """Create a normalization layer from a configuration dictionary."""

    if norm_cfg is None:
        return torch.nn.Identity()

    norm_type = norm_cfg.get("type", "BN").upper()
    eps = norm_cfg.get("eps", 1e-5)
    momentum = norm_cfg.get("momentum", 0.1)

    if norm_type == "BN":
        return torch.nn.BatchNorm2d(num_features, eps=eps, momentum=momentum)

    raise ValueError(f"Unsupported normalization type: {norm_type}")


class _ConvBNAct(torch.nn.Sequential):
    """Helper block performing convolution, normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        norm_cfg: Optional[dict],
    ) -> None:
        bias = norm_cfg is None
        layers = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=bias,
            )
        ]
        norm_layer = _build_norm_layer(out_channels, norm_cfg)
        if not isinstance(norm_layer, torch.nn.Identity):
            layers.append(norm_layer)
        layers.append(torch.nn.ReLU(inplace=True))
        super().__init__(*layers)


class MultiLevelNeck(torch.nn.Module):
    """Minimal implementation of the mmseg MultiLevelNeck in PyTorch."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        scales: Sequence[Union[int, float]],
    ) -> None:
        super().__init__()
        if len(in_channels) != len(scales):
            raise ValueError("`in_channels` and `scales` must have the same length")

        self.scales = list(scales)
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels]
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for feature, conv, scale in zip(inputs, self.convs, self.scales):
            x = conv(feature)
            if scale == 1:
                outputs.append(x)
                continue

            h = max(1, int(round(x.shape[-2] * scale)))
            w = max(1, int(round(x.shape[-1] * scale)))
            size = (h, w)

            if scale > 1:
                x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
            else:
                x = F.adaptive_avg_pool2d(x, output_size=size)
            outputs.append(x)

        return outputs


class PyramidPoolingModule(torch.nn.Module):
    """Pyramid pooling used by the UPerNet head."""

    def __init__(
        self,
        in_channels: int,
        channels: int,
        pool_scales: Iterable[int],
        norm_cfg: Optional[dict],
        align_corners: bool,
    ) -> None:
        super().__init__()
        self.align_corners = align_corners
        self.pool_scales = list(pool_scales)
        self.convs = torch.nn.ModuleList(
            [
                _ConvBNAct(
                    in_channels,
                    channels,
                    kernel_size=1,
                    padding=0,
                    norm_cfg=norm_cfg,
                )
                for _ in self.pool_scales
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [x]
        for scale, conv in zip(self.pool_scales, self.convs):
            pooled = F.adaptive_avg_pool2d(x, output_size=(scale, scale))
            out = conv(pooled)
            out = F.interpolate(
                out,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            outputs.append(out)

        return torch.cat(outputs, dim=1)


class UPerHead(torch.nn.Module):
    """PyTorch implementation of the UPerNet decode head."""

    def __init__(
        self,
        in_channels: Sequence[int],
        in_index: Sequence[int],
        pool_scales: Tuple[int, ...],
        channels: int,
        dropout_ratio: float,
        norm_cfg: Optional[dict],
        num_classes: int,
        align_corners: bool,
    ) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.in_index = list(in_index)
        self.channels = channels
        self.align_corners = align_corners
        self.dropout = (
            torch.nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else torch.nn.Identity()
        )
        self.classifier = torch.nn.Conv2d(channels, num_classes, kernel_size=1)

        self.lateral_convs = torch.nn.ModuleList(
            [
                _ConvBNAct(in_channels[i], channels, kernel_size=1, padding=0, norm_cfg=norm_cfg)
                for i in range(len(in_channels) - 1)
            ]
        )
        self.fpn_convs = torch.nn.ModuleList(
            [
                _ConvBNAct(channels, channels, kernel_size=3, padding=1, norm_cfg=norm_cfg)
                for _ in range(len(in_channels) - 1)
            ]
        )

        self.ppm = PyramidPoolingModule(
            in_channels[-1], channels, pool_scales, norm_cfg, align_corners
        )
        self.ppm_bottleneck = _ConvBNAct(
            in_channels[-1] + len(pool_scales) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
        )
        self.fpn_bottleneck = _ConvBNAct(
            len(in_channels) * channels,
            channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
        )

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        inputs = [inputs[i] for i in self.in_index]

        laterals = [
            lateral_conv(feature)
            for lateral_conv, feature in zip(self.lateral_convs, inputs[:-1])
        ]

        ppm_out = self.ppm(inputs[-1])
        laterals.append(self.ppm_bottleneck(ppm_out))

        for idx in range(len(laterals) - 1, 0, -1):
            prev_shape = laterals[idx - 1].shape[2:]
            laterals[idx - 1] = laterals[idx - 1] + F.interpolate(
                laterals[idx],
                size=prev_shape,
                mode="bilinear",
                align_corners=self.align_corners,
            )

        fpn_outs = [
            fpn_conv(lateral)
            for fpn_conv, lateral in zip(self.fpn_convs, laterals[:-1])
        ]
        fpn_outs.append(laterals[-1])

        output_size = fpn_outs[0].shape[2:]
        fpn_outs = [
            out
            if out.shape[2:] == output_size
            else F.interpolate(
                out,
                size=output_size,
                mode="bilinear",
                align_corners=self.align_corners,
            )
            for out in fpn_outs
        ]

        fusion = torch.cat(fpn_outs, dim=1)
        fusion = self.fpn_bottleneck(fusion)
        fusion = self.dropout(fusion)
        return self.classifier(fusion)


class FCNHead(torch.nn.Module):
    """Simple fully convolutional segmentation head."""

    def __init__(
        self,
        in_channels: int,
        in_index: int,
        channels: int,
        num_convs: int,
        concat_input: bool,
        dropout_ratio: float,
        num_classes: int,
        norm_cfg: Optional[dict],
        align_corners: bool,
    ) -> None:
        super().__init__()
        _ = concat_input  # kept for API compatibility
        self.in_index = in_index
        self.align_corners = align_corners

        layers = []
        last_channels = in_channels
        for _ in range(num_convs):
            layers.append(
                _ConvBNAct(
                    last_channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=norm_cfg,
                )
            )
            last_channels = channels

        self.convs = torch.nn.Sequential(*layers) if layers else torch.nn.Identity()
        self.dropout = (
            torch.nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else torch.nn.Identity()
        )
        self.classifier = torch.nn.Conv2d(last_channels, num_classes, kernel_size=1)

    def forward(self, inputs: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
        if isinstance(inputs, (tuple, list)):
            x = inputs[self.in_index]
        else:
            x = inputs

        x = self.convs(x)
        x = self.dropout(x)
        return self.classifier(x)


def resize(
    input: torch.Tensor,
    size: Optional[Tuple[int, int]] = None,
    scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    warning: Optional[bool] = None,
) -> torch.Tensor:
    """Wrapper around :func:`torch.nn.functional.interpolate` for API parity."""

    del warning  # maintained for compatibility with mmseg API
    return F.interpolate(
        input,
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )


class UPerNetWrapper(torch.nn.Module):
    def __init__(
        self, vit_backbone, feature_map_indices, num_classes=150, deepsup=False
    ):
        """
        Upernet-style wrapper around timm vit_large_patch16_224 with neck and decode_head from mmsegmentation
        """
        super().__init__()
        self.vit_backbone = vit_backbone
        self.deepsup = deepsup
        norm_cfg = dict(type="BN", eps=1e-6)
        self.neck = MultiLevelNeck(
            in_channels=[1024] * 4, out_channels=1024, scales=[4, 2, 1, 0.5]
        )
        self.head = UPerHead(
            in_channels=[1024] * 4,
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            num_classes=num_classes,
            align_corners=False,
        )
        if self.deepsup:
            self.aux_head = FCNHead(
                in_channels=1024,
                in_index=3,
                channels=256,
                num_convs=1,
                concat_input=False,
                dropout_ratio=0.1,
                num_classes=num_classes,
                norm_cfg=norm_cfg,
                align_corners=False,
            )
        self.feature_map_indices = feature_map_indices
        self.resize = resize

        assert max(feature_map_indices) <= len(
            self.vit_backbone.blocks
        ), "feature map index out of bounds"

    def forward_features(self, x):
        """return feature maps after specific transformer blocks"""
        x = self.vit_backbone.patch_embed(x)
        x = self.vit_backbone._pos_embed(x)
        x = self.vit_backbone.norm_pre(x)

        features = []
        for idx, block in enumerate(self.vit_backbone.blocks):
            x = block(x)

            if idx in self.feature_map_indices:
                features.append(x.clone())

        return features

    def forward_cd(self, x):
        assert x.shape[1] % 2 == 0
        mid = x.shape[1] // 2
        x1 = x[:, :mid]
        x2 = x[:, mid:]
        feat1 = self.forward_features(x1)
        feat2 = self.forward_features(x2)

        features = [torch.abs(f1 - f2) for f1, f2 in zip(feat1, feat2)]

        h = w = int(math.sqrt(features[0].shape[1] - 1))
        features = [
            einops.rearrange(z[:, 1:, :], "b (h w) d -> b d h w", h=h, w=w)
            for z in features
        ]
        features = self.neck(features)

        # `deepsup` prediction from intermediate feature map
        if self.deepsup:
            aux_map = self.aux_head(features)
            aux_map = self.resize(
                aux_map,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                warning=False,
            )

        # output feature map from extracted all feature maps
        output_map = self.head(features)
        output_map = self.resize(
            output_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            warning=False,
        )

        if self.deepsup:
            return output_map, aux_map

        return output_map

    def forward(self, x):
        features = self.forward_features(x)

        # remove cls token reshape into maps
        h = w = int(math.sqrt(features[0].shape[1] - 1))
        features = [
            einops.rearrange(z[:, 1:, :], "b (h w) d -> b d h w", h=h, w=w)
            for z in features
        ]
        features = self.neck(features)

        # `deepsup` prediction from intermediate feature map
        if self.deepsup:
            aux_map = self.aux_head(features)
            aux_map = self.resize(
                aux_map,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
                warning=False,
            )

        # output feature map from extracted all feature maps
        output_map = self.head(features)
        output_map = self.resize(
            output_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            warning=False,
        )

        if self.deepsup:
            return output_map, aux_map

        return output_map


class ViTWithFCNHead(torch.nn.Module):
    """ViT with fully-convolutional head for dense predictions."""

    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.deepsup = False
        norm_cfg = dict(type="BN", eps=1e-6)
        self.head = FCNHead(
            in_channels=1024,
            in_index=0,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
        )
        self.resize = resize

    def forward_features(self, x, feature_map_indices=[5, 11, 17, 23]):
        """return feature maps after specific transformer blocks"""
        x = self.backbone.patch_embed(x)
        x = self.backbone._pos_embed(x)
        x = self.backbone.norm_pre(x)

        features = []
        for idx, block in enumerate(self.backbone.blocks):
            x = block(x)

            if idx in feature_map_indices:
                features.append(x.clone())

        if len(features) == 1:
            return features[0]
        return features

    def forward(self, x):
        # features = self.backbone.forward_features(x)
        features = self.forward_features(x)
        if isinstance(features, list):
            h = w = int(math.sqrt(features[0].shape[1] - 1))
            features = [
                einops.rearrange(f[:, 1:, :], "b (h w) d -> b d h w", h=h, w=w)
                for f in features
            ]
            # note: this belongs below at output_map
            # features = [
            #     self.resize(
            #         f,
            #         size=(224, 224),
            #         mode="bilinear",
            #         align_corners=False,
            #         warning=False,
            #     )
            #     for f in features
            # ]
        else:
            h = w = int(math.sqrt(features.shape[1] - 1))
            features = einops.rearrange(
                features[:, 1:, :], "b (h w) d -> b d h w", h=h, w=w
            )
            features = [features]

        output_map = self.head(features)

        output_map = self.resize(
            output_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            warning=False,
        )

        return output_map

    def forward_cd(self, x):
        # x contains two images at different points in time
        # concatenated along the channel axis
        assert x.shape[1] % 2 == 0
        mid = x.shape[1] // 2
        x1 = x[:, :mid]
        x2 = x[:, mid:]

        feat1 = self.forward_features(x1)
        feat2 = self.forward_features(x2)

        if isinstance(feat1, list):
            feat = [torch.abs(f1 - f2) for f1, f2 in zip(feat1, feat2)]
            h = w = int(math.sqrt(feat[0].shape[1] - 1))
        else:
            feat = feat2 - feat1
            h = w = int(math.sqrt(feat.shape[1] - 1))

        if not isinstance(feat, list):
            feat = [feat]
        feat = [
            einops.rearrange(f[:, 1:, :], "b (h w) d -> b d h w", h=h, w=w)
            for f in feat
        ]

        output_map = self.head(feat)
        output_map = self.resize(
            output_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
            warning=False,
        )

        return output_map
