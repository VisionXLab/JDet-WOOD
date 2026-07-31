# TEED: is a Tiny but Efficient Edge Detection, it comes from the LDC-B3
# with a Slightly modification
# Jittor port of Point2RBox-v3's TED edge detector.
# state_dict 键名与 torch 版逐一对应，权重经 tools/convert_ted_weights.py 转成 ted.pkl 加载。
import jittor as jt
from jittor import nn


def smish(x):
    """mish(x) = x * tanh(ln(1 + sigmoid(x)))"""
    return x * jt.tanh(jt.log(1 + jt.sigmoid(x)))


class Smish(nn.Module):

    def execute(self, x):
        return smish(x)


class CoFusion(nn.Module):
    # from LDC

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
    # TEDv14-3

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
        self.PSconv1 = nn.PixelShuffle(1)
        self.DWconv2 = nn.Conv2d(24, 24 * 1, kernel_size=3,
                                 stride=1, padding=1, groups=24)
        self.AF = Smish()

    def execute(self, x):
        attn = self.PSconv1(self.DWconv1(self.AF(x)))
        attn2 = self.PSconv1(self.DWconv2(self.AF(attn)))
        return smish(((attn2 + attn).sum(1)).unsqueeze(1))


class _DenseLayer(nn.Module):
    """torch 版是 nn.Sequential 子类（conv1→smish1→conv2），键名保持一致。"""

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
        self.num_layers = num_layers
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            setattr(self, 'denselayer%d' % (i + 1), layer)
            input_features = out_features

    def execute(self, x):
        for i in range(self.num_layers):
            x = getattr(self, 'denselayer%d' % (i + 1))(x)
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
            layers.append(nn.ConvTranspose2d(
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
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=True)
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
        self.conv1 = nn.Conv2d(in_features, mid_features, 3, padding=1, stride=stride)
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
    """Tiny and Efficient Edge Detector（权重加载：ted.pkl，numpy dict）。"""

    def __init__(self):
        super(TED, self).__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(1, 32, 48)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.side_1 = SingleConvBlock(16, 32, 2)
        self.pre_dense_3 = SingleConvBlock(32, 48, 1)

        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(48, 2)

        self.block_cat = DoubleFusion(3, 3)

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

        block_cat = jt.concat(results, dim=1)
        block_cat = self.block_cat(block_cat)

        results.append(block_cat)
        return results
