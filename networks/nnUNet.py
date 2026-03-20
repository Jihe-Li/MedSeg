import torch
import numpy as np

from typing import Union, Type, List, Tuple
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from dynamic_network_architectures.architectures.abstract_arch import AbstractDynamicNetworkArchitectures
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp


class ResidualEncoderUNet(AbstractDynamicNetworkArchitectures):
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        num_classes: int,
        n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        deep_supervision: bool = False,
        block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
        bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
        stem_channels: int = None,
    ):
        super().__init__()

        self.key_to_encoder = "encoder.stages"
        self.key_to_stem = "encoder.stem"
        self.keys_to_in_proj = ("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0")

        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, (
            "n_blocks_per_stage must have as many entries as we have "
            f"resolution stages. here: {n_stages}. "
            f"n_blocks_per_stage: {n_blocks_per_stage}"
        )
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), (
            "n_conv_per_stage_decoder must have one less entries "
            f"as we have resolution stages. here: {n_stages} "
            f"stages, so it should have {n_stages - 1} entries. "
            f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        )
        self.encoder = ResidualEncoder(
            input_channels,
            n_stages,
            features_per_stage,
            conv_op,
            kernel_sizes,
            strides,
            n_blocks_per_stage,
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs,
            block,
            bottleneck_channels,
            return_skips=True,
            disable_default_stem=False,
            stem_channels=stem_channels,
        )
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), (
            "just give the image size without color/feature channels or "
            "batch channel. Do not give input_size=(b, c, x, y(, z)). "
            "Give input_size=(x, y(, z))!"
        )
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(
            input_size
        )

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)

class UNetDecoder(nn.Module):
    def __init__(self,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None
                 ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs


        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, 2 * input_features_skip, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing

        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = encoder.conv_op(encoder.output_channels[0], num_classes, 1, 1, 0, bias=True)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.stages[s](x)
            lres_input = x
        self.seg_layers.to(torch.float32)
        seg_output = self.seg_layers(x.float())
        return seg_output

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output


def get_keys_to_pop(pretrained_weights, prefixes):
    return [k for k in pretrained_weights.keys() if any(k.startswith(p) for p in prefixes)]

def get_NNSSL(config):
    ckpt_path = config.ckpt_path
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    arch_config = ckpt['init_args']['plan']['configurations']['3d_fullres']
    arch_kwargs = {'n_stages': 6,
        'features_per_stage': [32, 64, 128, 256, 320, 320],
        'conv_op': nn.Conv3d,
        'kernel_sizes': arch_config['conv_kernel_sizes'],
        'strides': arch_config['pool_op_kernel_sizes'],
        'n_blocks_per_stage': arch_config['n_conv_per_stage_encoder'],
        'n_conv_per_stage_decoder': arch_config['n_conv_per_stage_decoder'],
        'conv_bias': True,
        'norm_op': nn.InstanceNorm3d,
        'norm_op_kwargs': {"eps": 1e-5, "affine": True},
        'nonlin': nn.LeakyReLU,
        'nonlin_kwargs': {"inplace": True},
        'dropout_op': None,
        'dropout_op_kwargs': None}

    arch_kwargs['input_channels'] = config.input_channels
    arch_kwargs['num_classes'] = config.num_classes
    model = ResidualEncoderUNet(**arch_kwargs)
    if config.load_weights:
        pretrained_weights = ckpt['network_weights']
        prefixes = ['decoder.seg_layers']
        keys_to_pop = get_keys_to_pop(pretrained_weights, prefixes)
        for key in keys_to_pop:
            pretrained_weights.pop(key)

        missing_keys, unexpected_keys = model.load_state_dict(pretrained_weights, strict=False)
        print("Missing parameter keys:", missing_keys)
        print("Unexpected parameter keys:", unexpected_keys)
    else:
        print("Without loading pretrained weights for model.")
    return model

def get_nnInteractive(config):
    ckpt_path = config.ckpt_path
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    arch_kwargs = ckpt['init_args']['plans']['configurations']['3d_fullres_ps192']['architecture']['arch_kwargs']
    arch_kwargs['input_channels'] = config.input_channels
    arch_kwargs['num_classes'] = config.num_classes
    for key, value in arch_kwargs.items():
        if isinstance(value, str):
            arch_kwargs[key] = eval(value)

    model = ResidualEncoderUNet(**arch_kwargs)
    if config.load_weights:
        pretrained_weights = ckpt['network_weights']
        prefixes = ['decoder.seg_layers']
        keys_to_pop = get_keys_to_pop(pretrained_weights, prefixes)
        for key in keys_to_pop:
            pretrained_weights.pop(key)
        keys_to_modify = [
            'encoder.stem.convs.0.conv.weight',
            'encoder.stem.convs.0.all_modules.0.weight',
            'decoder.encoder.stem.convs.0.conv.weight',
            'decoder.encoder.stem.convs.0.all_modules.0.weight'
        ]
        for key in keys_to_modify:
            original_weights = pretrained_weights[key]
            sliced_weights_ch0 = original_weights[:, 0:1, :, :, :]
            pretrained_weights[key] = sliced_weights_ch0

        missing_keys, unexpected_keys = model.load_state_dict(pretrained_weights, strict=False)
        print("Missing parameter keys:", missing_keys)
        print("Unexpected parameter keys:", unexpected_keys)
    else:
        print("Without loading pretrained weights for model.")

    return model


if __name__ == "__main__":

    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from utils import io
    from types import SimpleNamespace

    config = io.load_yaml('configs/seg_networks/nnInteractive.yaml')
    config = SimpleNamespace(**config)

    ckpt_path = config.ckpt_path
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    arch_kwargs = ckpt['init_args']['plans']['configurations']['3d_fullres_ps192']['architecture']['arch_kwargs']
    arch_kwargs['input_channels'] = config.input_channels
    arch_kwargs['num_classes'] = config.num_classes
    for key, value in arch_kwargs.items():
        if isinstance(value, str):
            arch_kwargs[key] = eval(value)

    model = ResidualEncoderUNet(**arch_kwargs).cuda()
    pretrained_weights = ckpt['network_weights']
    prefixes = ['decoder.seg_layers']
    keys_to_pop = get_keys_to_pop(pretrained_weights, prefixes)
    for key in keys_to_pop:
        pretrained_weights.pop(key)
    keys_to_modify = [
        'encoder.stem.convs.0.conv.weight',
        'encoder.stem.convs.0.all_modules.0.weight',
        'decoder.encoder.stem.convs.0.conv.weight',
        'decoder.encoder.stem.convs.0.all_modules.0.weight'
    ]
    for key in keys_to_modify:
        original_weights = pretrained_weights[key]
        sliced_weights_ch0 = original_weights[:, 0:1, :, :, :]
        pretrained_weights[key] = sliced_weights_ch0

    missing_keys, unexpected_keys = model.load_state_dict(pretrained_weights, strict=False)
    print("Missing parameter keys:", missing_keys)
    print("Unexpected parameter keys:", unexpected_keys)

    img_ten = torch.randn(1, 1, 160, 160, 160).cuda()
    out = model(img_ten)
    breakpoint()
