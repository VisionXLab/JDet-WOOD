# TED (Tiny and Efficient Edge Detector), ported to Jittor from
# Point2RBox-v3 third_parties/ted/ted.py (upstream: TEED / LDC-B3).
# Structure and parameter names are kept identical to the PyTorch version so
# that weights converted by tools/convert_torch_weights.py load directly.
# NOTE: smish here is x * tanh(log(1 + sigmoid(x))) — this matches the
# upstream implementation exactly and is NOT the standard mish/softplus form.

import jittor as jt
from jittor import nn


def smish(x):
    return x * jt.tanh(jt.log(1 + jt.sigmoid(x)))


if hasattr(nn, 'PixelShuffle'):
    PixelShuffle = nn.PixelShuffle
else:
    class PixelShuffle(nn.Module):
        # Fallback for Jittor builds without nn.PixelShuffle. TED only uses
        # upscale_factor=1 (identity), but the general form is kept.

        def __init__(self, upscale_factor):
            super(PixelShuffle, self).__init__()
            self.upscale_factor = upscale_factor

        def execute(self, x):
            r = self.upscale_factor
            if r == 1:
                return x
            n, c, h, w = x.shape
            x = x.reshape(n, c // (r * r), r, r, h, w)
            x = x.permute(0, 1, 4, 2, 5, 3)
            return x.reshape(n, c // (r * r), h * r, w * r)


class Smish(nn.Module):

    def execute(self, x):
        return smish(x)


class CoFusion(nn.Module):
    # from LDC; kept for structural completeness (not used by TED.execute)

    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, out_ch, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.norm_layer1 = nn.GroupNorm(4, 32)

    def execute(self, x):
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = nn.softmax(self.conv3(attn), dim=1)
        return ((x * attn).sum(1)).unsqueeze(1)


class CoFusion2(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(CoFusion2, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, out_ch, kernel_size=3, stride=1, padding=1)
        self.smish = Smish()

    def execute(self, x):
        attn = self.conv1(self.smish(x))
        attn = self.conv3(self.smish(attn))
        return ((x * attn).sum(1)).unsqueeze(1)


class DoubleFusion(nn.Module):
    # TED fusion before the final edge map prediction

    def __init__(self, in_ch, out_ch):
        super(DoubleFusion, self).__init__()
        self.DWconv1 = nn.Conv2d(in_ch, in_ch * 8, kernel_size=3,
                                 stride=1, padding=1, groups=in_ch)
        self.PSconv1 = PixelShuffle(1)
        self.DWconv2 = nn.Conv2d(24, 24 * 1, kernel_size=3,
                                 stride=1, padding=1, groups=24)
        self.AF = Smish()

    def execute(self, x):
        attn = self.PSconv1(self.DWconv1(self.AF(x)))
        attn2 = self.PSconv1(self.DWconv2(self.AF(attn)))
        return smish(((attn2 + attn).sum(1)).unsqueeze(1))


class _DenseLayer(nn.Module):
    # Upstream subclasses nn.Sequential with submodules named
    # conv1/smish1/conv2; plain attributes here reproduce the same
    # state_dict keys. conv1 has padding=2 while conv2 has padding=0 —
    # asymmetric on purpose, copied verbatim.

    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()
        self.conv1 = nn.Conv2d(input_features, out_features,
                               kernel_size=3, stride=1, padding=2, bias=True)
        self.smish1 = Smish()
        self.conv2 = nn.Conv2d(out_features, out_features,
                               kernel_size=3, stride=1, bias=True)

    def execute(self, x):
        x1, x2 = x
        new_features = self.conv2(self.smish1(self.conv1(smish(x1))))
        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Module):

    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        self.layer_names = []
        for i in range(num_layers):
            name = 'denselayer%d' % (i + 1)
            setattr(self, name, _DenseLayer(input_features, out_features))
            self.layer_names.append(name)
            input_features = out_features

    def execute(self, x):
        for name in self.layer_names:
            x = getattr(self, name)(x)
        return x


class UpConvBlock(nn.Module):

    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16
        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads = [0, 0, 1, 3, 7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(Smish())
            layers.append(nn.ConvTranspose(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def execute(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):

    def __init__(self, in_features, out_features, stride, use_ac=False):
        super(SingleConvBlock, self).__init__()
        self.use_ac = use_ac
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride,
                              bias=True)
        if self.use_ac:
            self.smish = Smish()

    def execute(self, x):
        x = self.conv(x)
        if self.use_ac:
            return self.smish(x)
        return x


class DoubleConvBlock(nn.Module):

    def __init__(self, in_features, mid_features,
                 out_features=None,
                 stride=1,
                 use_act=True):
        super(DoubleConvBlock, self).__init__()
        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features,
                               3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.smish = Smish()

    def execute(self, x):
        x = self.conv1(x)
        x = self.smish(x)
        x = self.conv2(x)
        if self.use_act:
            x = self.smish(x)
        return x


class TED(nn.Module):
    """Tiny and Efficient Edge Detector (Jittor port).

    Inference-only in Point2RBox-v3: weights are loaded from the converted
    upstream checkpoint and frozen, so the torch-side weight_init is omitted.
    """

    def __init__(self):
        super(TED, self).__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(1, 32, 48)

        self.maxpool = nn.Pool(kernel_size=3, stride=2, padding=1, op='maximum')

        self.side_1 = SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = SingleConvBlock(32, 48, 1)

        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2)

        self.block_cat = DoubleFusion(3, 3)

    def slice(self, tensor, slice_shape):
        t_shape = tensor.shape
        img_h, img_w = slice_shape
        if img_w != t_shape[-1] or img_h != t_shape[2]:
            new_tensor = nn.interpolate(
                tensor, size=(img_h, img_w), mode='bicubic', align_corners=False)
        else:
            new_tensor = tensor
        return new_tensor

    def resize_input(self, tensor):
        t_shape = tensor.shape
        if t_shape[2] % 8 != 0 or t_shape[3] % 8 != 0:
            img_w = ((t_shape[3] // 8) + 1) * 8
            img_h = ((t_shape[2] // 8) + 1) * 8
            new_tensor = nn.interpolate(
                tensor, size=(img_h, img_w), mode='bicubic', align_corners=False)
        else:
            new_tensor = tensor
        return new_tensor

    def execute(self, x, single_test=False):
        assert x.ndim == 4, x.shape

        # Block 1
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)

        results = [out_1, out_2, out_3]

        # concatenate multiscale outputs
        block_cat = jt.concat(results, dim=1)  # Bx3xHxW
        block_cat = self.block_cat(block_cat)  # Bx1xHxW DoubleFusion

        results.append(block_cat)
        return results
