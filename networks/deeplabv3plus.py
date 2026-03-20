"""
3D DeepLabv3+ for medical image segmentation/regression.

This implementation combines the modular structure from the previous 2D file
with the configurable DeepLabv3+ design patterns used in `tets.py`:
- Configurable ResNet backbone depth (34/50/101)
- Configurable output stride (8/16)
- 3D ASPP and 3D decoder
- Configurable normalization: BatchNorm / InstanceNorm / GroupNorm

Important:
The global pooling branch inside ASPP always uses GroupNorm. InstanceNorm cannot
operate on pooled 1x1x1 feature maps during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_group_count(num_channels, desired_groups):
  groups = min(desired_groups, num_channels)
  while num_channels % groups != 0 and groups > 1:
    groups -= 1
  return max(groups, 1)


def get_norm_3d(norm_type, num_channels, num_groups=8, force_group_norm=False):
  norm_key = norm_type.lower()
  if force_group_norm:
    groups = _resolve_group_count(num_channels, num_groups)
    return nn.GroupNorm(groups, num_channels)
  if norm_key in ("batch", "batchnorm", "bn"):
    return nn.BatchNorm3d(num_channels)
  if norm_key in ("instance", "instancenorm", "in"):
    return nn.InstanceNorm3d(num_channels, affine=True)
  if norm_key in ("group", "groupnorm", "gn"):
    groups = _resolve_group_count(num_channels, num_groups)
    return nn.GroupNorm(groups, num_channels)
  raise ValueError(f"Unsupported norm_type: {norm_type}")


class BasicBlock3D(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=(1, 1, 1), rate=1, downsample=None,
               norm_type="batch", norm_num_groups=8):
    super().__init__()
    dilation = (rate, rate, rate)
    padding = dilation
    self.conv1 = nn.Conv3d(
      inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
    )
    self.bn1 = get_norm_3d(norm_type, planes, norm_num_groups)
    self.conv2 = nn.Conv3d(
      planes, planes, kernel_size=3, stride=1, dilation=dilation, padding=padding, bias=False
    )
    self.bn2 = get_norm_3d(norm_type, planes, norm_num_groups)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample

  def forward(self, x):
    identity = x
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    if self.downsample is not None:
      identity = self.downsample(x)
    out = self.relu(out + identity)
    return out


class Bottleneck3D(nn.Module):
  expansion = 4

  def __init__(self, inplanes, planes, stride=(1, 1, 1), rate=1, downsample=None,
               norm_type="batch", norm_num_groups=8):
    super().__init__()
    dilation = (rate, rate, rate)
    padding = dilation
    self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = get_norm_3d(norm_type, planes, norm_num_groups)
    self.conv2 = nn.Conv3d(
      planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=padding, bias=False
    )
    self.bn2 = get_norm_3d(norm_type, planes, norm_num_groups)
    self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
    self.bn3 = get_norm_3d(norm_type, planes * self.expansion, norm_num_groups)
    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample

  def forward(self, x):
    identity = x
    out = self.relu(self.bn1(self.conv1(x)))
    out = self.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    if self.downsample is not None:
      identity = self.downsample(x)
    out = self.relu(out + identity)
    return out


class ResNet3DEncoder(nn.Module):
  def __init__(self, in_channels, block, layers, output_stride=16, norm_type="batch", norm_num_groups=8):
    super().__init__()
    self.inplanes = 64
    self.norm_type = norm_type
    self.norm_num_groups = norm_num_groups

    if output_stride == 16:
      strides = [(1, 1, 1), (2, 2, 2), (2, 2, 2), (1, 1, 1)]
      rates = [1, 1, 1, 2]
      mg_blocks = [1, 2, 4]
    elif output_stride == 8:
      strides = [(1, 1, 1), (2, 2, 2), (1, 1, 1), (1, 1, 1)]
      rates = [1, 1, 2, 2]
      mg_blocks = [1, 2, 1]
    else:
      raise ValueError(f"Unsupported output_stride: {output_stride}")

    self.conv1 = nn.Conv3d(
      in_channels, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False
    )
    self.bn1 = get_norm_3d(norm_type, 64, norm_num_groups)
    self.relu = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

    self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], rate=rates[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], rate=rates[1])
    self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], rate=rates[2])
    self.layer4 = self._make_mg_unit(block, 512, blocks=mg_blocks, stride=strides[3], rate=rates[3])

    self._init_weight()
    self.low_level_channels = 64 * block.expansion
    self.high_level_channels = 512 * block.expansion

  def _make_layer(self, block, planes, blocks, stride=(1, 1, 1), rate=1):
    downsample = None
    if stride != (1, 1, 1) or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
        get_norm_3d(self.norm_type, planes * block.expansion, self.norm_num_groups),
      )
    layers = [
      block(
        self.inplanes, planes, stride=stride, rate=rate, downsample=downsample,
        norm_type=self.norm_type, norm_num_groups=self.norm_num_groups
      )
    ]
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(
        block(
          self.inplanes, planes, stride=(1, 1, 1), rate=rate,
          norm_type=self.norm_type, norm_num_groups=self.norm_num_groups
        )
      )
    return nn.Sequential(*layers)

  def _make_mg_unit(self, block, planes, blocks, stride=(1, 1, 1), rate=1):
    downsample = None
    if stride != (1, 1, 1) or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
        get_norm_3d(self.norm_type, planes * block.expansion, self.norm_num_groups),
      )
    layers = [
      block(
        self.inplanes, planes, stride=stride, rate=blocks[0] * rate, downsample=downsample,
        norm_type=self.norm_type, norm_num_groups=self.norm_num_groups
      )
    ]
    self.inplanes = planes * block.expansion
    for idx in range(1, len(blocks)):
      layers.append(
        block(
          self.inplanes, planes, stride=(1, 1, 1), rate=blocks[idx] * rate,
          norm_type=self.norm_type, norm_num_groups=self.norm_num_groups
        )
      )
    return nn.Sequential(*layers)

  def _init_weight(self):
    for module in self.modules():
      if isinstance(module, nn.Conv3d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
      elif isinstance(module, nn.GroupNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
      elif isinstance(module, nn.InstanceNorm3d) and module.affine:
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

  def forward(self, x):
    x = self.relu(self.bn1(self.conv1(x)))
    x = self.maxpool(x)
    low_level = self.layer1(x)
    x = self.layer2(low_level)
    x = self.layer3(x)
    high_level = self.layer4(x)
    return low_level, high_level


class AtrousSeparableConv3D(nn.Module):
  def __init__(self, in_channels, out_channels, dilation, norm_type="batch", norm_num_groups=8):
    super().__init__()
    dilation_3d = (dilation, dilation, dilation)
    padding_3d = dilation_3d
    self.depthwise = nn.Conv3d(
      in_channels, in_channels, kernel_size=3, dilation=dilation_3d,
      padding=padding_3d, groups=in_channels, bias=False
    )
    self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
    self.norm = get_norm_3d(norm_type, out_channels, norm_num_groups)
    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    x = self.depthwise(x)
    x = self.pointwise(x)
    x = self.norm(x)
    return self.relu(x)


class ASPPModule3D(nn.Module):
  def __init__(self, in_channels, out_channels=256, atrous_rates=(6, 12, 18),
               norm_type="batch", norm_num_groups=8):
    super().__init__()
    self.branches = nn.ModuleList()
    self.branches.append(
      nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
        get_norm_3d(norm_type, out_channels, norm_num_groups),
        nn.ReLU(inplace=True),
      )
    )
    for rate in atrous_rates:
      self.branches.append(
        AtrousSeparableConv3D(
          in_channels, out_channels, dilation=rate,
          norm_type=norm_type, norm_num_groups=norm_num_groups
        )
      )

    # Force GroupNorm here: pooled features become (N, C, 1, 1, 1).
    self.global_pool = nn.Sequential(
      nn.AdaptiveAvgPool3d(1),
      nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
      get_norm_3d(norm_type, out_channels, norm_num_groups, force_group_norm=True),
      nn.ReLU(inplace=True),
    )

    total_channels = out_channels * (len(atrous_rates) + 2)
    self.project = nn.Sequential(
      nn.Conv3d(total_channels, out_channels, kernel_size=1, bias=False),
      get_norm_3d(norm_type, out_channels, norm_num_groups),
      nn.ReLU(inplace=True),
      nn.Dropout3d(0.5),
    )

  def forward(self, x):
    outputs = [branch(x) for branch in self.branches]
    pooled = self.global_pool(x)
    pooled = F.interpolate(pooled, size=x.shape[2:], mode="trilinear", align_corners=False)
    outputs.append(pooled)
    return self.project(torch.cat(outputs, dim=1))


class DecoderModule3D(nn.Module):
  def __init__(self, low_level_channels, num_classes=1, norm_type="batch", norm_num_groups=8):
    super().__init__()
    self.low_level_proj = nn.Sequential(
      nn.Conv3d(low_level_channels, 48, kernel_size=1, bias=False),
      get_norm_3d(norm_type, 48, norm_num_groups),
      nn.ReLU(inplace=True),
    )
    self.decode = nn.Sequential(
      nn.Conv3d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
      get_norm_3d(norm_type, 256, norm_num_groups),
      nn.ReLU(inplace=True),
      nn.Dropout3d(0.5),
      nn.Conv3d(256, 256, kernel_size=3, padding=1, bias=False),
      get_norm_3d(norm_type, 256, norm_num_groups),
      nn.ReLU(inplace=True),
      nn.Dropout3d(0.1),
    )
    self.classifier = nn.Conv3d(256, num_classes, kernel_size=1)

  def forward(self, encoder_features, low_level_features):
    low_level_features = self.low_level_proj(low_level_features)
    encoder_features = F.interpolate(
      encoder_features, size=low_level_features.shape[2:], mode="trilinear", align_corners=False
    )
    x = torch.cat([encoder_features, low_level_features], dim=1)
    x = self.decode(x)
    return self.classifier(x)


def _get_backbone_spec(backbone):
  key = backbone.lower()
  if key == "resnet34":
    return BasicBlock3D, [3, 4, 6, 3]
  if key == "resnet50":
    return Bottleneck3D, [3, 4, 6, 3]
  if key == "resnet101":
    return Bottleneck3D, [3, 4, 23, 3]
  raise ValueError(f"Unsupported backbone: {backbone}")


class DeepLabv3Plus3D(nn.Module):
  """
  3D DeepLabv3+ with configurable normalization.

  Args:
    in_channels: input channels for 3D volume, shape (N, C, D, H, W).
    num_classes: output channels.
    backbone: "resnet34" | "resnet50" | "resnet101".
    output_stride: 8 or 16.
    norm_type: "instance" | "batch" | "group".
    norm_num_groups: number of groups for GroupNorm.
  """

  def __init__(
    self,
    in_channels=1,
    num_classes=1,
    backbone="resnet50",
    output_stride=16,
    norm_type="batch",
    norm_num_groups=8,
  ):
    super().__init__()
    block, layers = _get_backbone_spec(backbone)
    self.encoder = ResNet3DEncoder(
      in_channels=in_channels,
      block=block,
      layers=layers,
      output_stride=output_stride,
      norm_type=norm_type,
      norm_num_groups=norm_num_groups,
    )
    if output_stride == 16:
      aspp_rates = (6, 12, 18)
    else:
      aspp_rates = (12, 24, 36)

    self.aspp = ASPPModule3D(
      in_channels=self.encoder.high_level_channels,
      out_channels=256,
      atrous_rates=aspp_rates,
      norm_type=norm_type,
      norm_num_groups=norm_num_groups,
    )
    self.decoder = DecoderModule3D(
      low_level_channels=self.encoder.low_level_channels,
      num_classes=num_classes,
      norm_type=norm_type,
      norm_num_groups=norm_num_groups,
    )

  def forward(self, x):
    input_shape = x.shape[2:]
    low_level_features, high_level_features = self.encoder(x)
    x = self.aspp(high_level_features)
    x = self.decoder(x, low_level_features)
    breakpoint()  
    x = F.interpolate(x, size=input_shape, mode="trilinear", align_corners=False)
    return F.relu(x)


def build_deeplabv3plus(config=None, structure_selector=None, out_channels=1):
  config = config or {}
  model_cfg = config.get("deeplabv3plus", {})

  in_channels = model_cfg.get("in_channels", 1)
  if structure_selector is not None and hasattr(structure_selector, "total_channels"):
    in_channels = structure_selector.total_channels

  model = DeepLabv3Plus3D(
    in_channels=in_channels,
    num_classes=model_cfg.get("num_classes", out_channels),
    backbone=model_cfg.get("backbone", "resnet50"),
    output_stride=model_cfg.get("os", model_cfg.get("output_stride", 16)),
    norm_type=model_cfg.get("norm_type", "batch"),
    norm_num_groups=model_cfg.get("norm_num_groups", 8),
  )
  return model


def get_DeepLabv3Plus(config=None, structure_selector=None, out_channels=1):
  return build_deeplabv3plus(config=config, structure_selector=structure_selector, out_channels=out_channels)


if __name__ == "__main__":
  example_cfg = {"deeplabv3plus": {"in_channels": 1, "num_classes": 1, "norm_type": "instance"}}
  model = build_deeplabv3plus(config=example_cfg)
  inputs = torch.randn(1, 1, 128, 128, 128)
  outputs = model(inputs)
  print(outputs.shape)
