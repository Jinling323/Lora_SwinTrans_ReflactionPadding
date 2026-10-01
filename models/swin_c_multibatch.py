import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models import Swin_T_Weights, swin_t
from torchvision.models.swin_transformer import SwinTransformer
import os
import re

from .transformer_cosine_multibatch import (
    TransformerEncoder,
    TransformerEncoderLayer,
)
from .swin_lora import add_swin_qv_lora


__all__ = ["swin_t_trans", "swin_l_trans"]


def _load_official_swin_large(backbone, path):
    """Map the official Swin-L 22K state dict onto torchvision's Swin-L."""
    if not os.path.isfile(path):
        raise FileNotFoundError('Swin-L pretrained checkpoint not found: {}'.format(path))
    try:
        checkpoint = torch.load(path, map_location='cpu', mmap=True)
    except TypeError:
        checkpoint = torch.load(path, map_location='cpu')
    source = checkpoint.get('model', checkpoint)
    expected = backbone.state_dict()
    mapped = {}
    for key, value in source.items():
        target = None
        if key.startswith('patch_embed.proj.'):
            target = key.replace('patch_embed.proj.', 'features.0.0.', 1)
        elif key.startswith('patch_embed.norm.'):
            target = key.replace('patch_embed.norm.', 'features.0.2.', 1)
        elif key.startswith('norm.'):
            target = key
        else:
            block = re.fullmatch(r'layers\.(\d+)\.blocks\.(\d+)\.(.*)', key)
            downsample = re.fullmatch(r'layers\.(\d+)\.downsample\.(.*)', key)
            if block:
                stage, index, suffix = block.groups()
                suffix = suffix.replace('mlp.fc1.', 'mlp.0.')
                suffix = suffix.replace('mlp.fc2.', 'mlp.3.')
                target = 'features.{}.{}.{}'.format(2 * int(stage) + 1, index, suffix)
            elif downsample:
                stage, suffix = downsample.groups()
                target = 'features.{}.{}'.format(2 * int(stage) + 2, suffix)
        # torchvision generates the same relative-position indices itself;
        # the source also stores resolution-specific attention masks.
        if target is None or target.endswith(('relative_position_index', 'attn_mask')):
            continue
        if target not in expected:
            raise ValueError('Unrecognized Swin-L checkpoint parameter: {}'.format(key))
        if value.shape != expected[target].shape:
            raise ValueError('Swin-L checkpoint shape mismatch: {}'.format(key))
        mapped[target] = value
    missing = [key for key in expected if key.startswith(('features.', 'norm.'))
               and key not in mapped and not key.endswith('relative_position_index')]
    if missing:
        raise ValueError('Incomplete Swin-L checkpoint; first missing key: {}'.format(missing[0]))
    backbone.load_state_dict(mapped, strict=False)


class SwinTransMultiBatch(nn.Module):
    """Swin backbone followed by the original multibatch MAN head."""

    def __init__(self, pretrained=True, variant='large', pretrained_path=None):
        super().__init__()
        if variant == 'large':
            backbone = SwinTransformer(
                patch_size=[4, 4], embed_dim=192, depths=[2, 2, 18, 2],
                num_heads=[6, 12, 24, 48], window_size=[12, 12])
            if pretrained:
                path = (pretrained_path or
                        'pre_models/swin_large_patch4_window12_384_22k.pth')
                _load_official_swin_large(backbone, path)
            output_channels = 1536
        elif variant == 'tiny':
            weights = Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = swin_t(weights=weights)
            output_channels = 768
        else:
            raise ValueError('Unknown Swin variant: {}'.format(variant))
        self.variant = variant
        self.backbone_features = backbone.features
        self.backbone_norm = backbone.norm

        # Both backbones end at stride 32; the existing MAN encoder uses 512 channels.
        self.feature_adapter = nn.Sequential(
            nn.Conv2d(output_channels, 512, kernel_size=1),
            nn.GELU(),
        )

        d_model = 512
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=2,
            dim_feedforward=2048,
            dropout=0.1,
            activation="relu",
            normalize_before=False,
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=2, norm=None)

        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, x):
        input_h, input_w = x.shape[-2:]
        density_size = (input_h // 16, input_w // 16)

        # torchvision Swin features and norm use channels-last tensors.
        x = self.backbone_features(x)
        x = self.backbone_norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.feature_adapter(x)

        batch_size, channels, height, width = x.shape
        tokens = x.flatten(2).permute(2, 0, 1)
        tokens, consistency_features = self.encoder(tokens, (height, width))
        x = tokens.permute(1, 2, 0).reshape(
            batch_size, channels, height, width
        )

        x = F.interpolate(
            x, size=density_size, mode="bilinear", align_corners=False
        )
        density = self.reg_layer_0(x)
        return torch.relu(density), consistency_features

    def enable_lora(self, rank=4, alpha=4.0):
        """Freeze the clean baseline and adapt all backbone attention Q/V."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return add_swin_qv_lora(self.backbone_features, rank, alpha)


def swin_t_trans(pretrained=True, pretrained_path=None):
    return SwinTransMultiBatch(pretrained=pretrained, variant='tiny')


def swin_l_trans(pretrained=True, pretrained_path=None):
    return SwinTransMultiBatch(
        pretrained=pretrained, variant='large', pretrained_path=pretrained_path)
