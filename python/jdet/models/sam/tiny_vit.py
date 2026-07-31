# Ported from MobileSAM `modeling/tiny_vit_sam.py`.
# (TinyViT, Microsoft, adapted from LeViT / Swin) for Point2RBox-v3-jittor.
# Inference-only: vit_t (MobileSAM) is the only supported configuration.
#
# Faithfulness notes:
#   - All module attribute names / Sequential orderings match the PyTorch
#     version exactly so converted state_dict keys line up 1:1
#     (e.g. `layers.1.blocks.0.attn.qkv.weight`, `patch_embed.seq.0.c.weight`).
#   - Conv2d_BN is a Module with children named 'c' and 'bn' (torch uses a
#     named Sequential; key paths are identical).
#   - `attention_biases` is a trainable parameter (in checkpoint);
#     `attention_bias_idxs` is a persistent=False buffer in torch (NOT in the
#     checkpoint), so here it is kept as a plain numpy attribute and a
#     private jt.Var cache `_bias_idx_flat` (underscore names are not
#     matched by load_parameters).
#   - torch's Attention.train()/eval() 'ab' caching is dropped: execute
#     always gathers from attention_biases, which is numerically identical.
#   - set_layer_lr_decay / _init_weights / trunc_normal_ are training-time
#     concerns; weights come from the converted checkpoint. Omitted.
#   - torch.utils.checkpoint branches removed; use_checkpoint arg kept and
#     ignored.
#   - Conv2d_BN.fuse() omitted (inference works unfused; parity tests must
#     run against the unfused torch model, which is the default).
#   - classifier head (norm_head/head) kept for state_dict compatibility
#     even though forward never uses it.
#   - torch `.transpose(a, b)` is rewritten as `.permute(...)` with the full
#     dim order — Jittor's Var.transpose takes a whole permutation and does
#     NOT mean "swap two dims".
#
# UNVERIFIED-API (check once env is ready):
#   - nn.BatchNorm(b) runs per-channel over NCHW with attrs
#     weight/bias/running_mean/running_var (JDet base uses it this way)
#   - nn.ModuleList exists; children get numeric names ("0", "1", ...)
#   - nn.Sequential numeric key naming matches torch ("seq.0.c.weight")
#   - jt.nn.pad(x, (0, 0, 0, pad_r, 0, pad_b)) pads last-dim-first like torch
#   - fancy indexing `param[:, flat_idx]` gathers along dim 1
#   - nn.Identity exists (fallback provided)
#   - x.flatten(2) flattens dims 2..end

import itertools
import numpy as np

import jittor as jt
from jittor import nn


def to_2tuple(x):
    if isinstance(x, (tuple, list)):
        assert len(x) == 2
        return tuple(x)
    return (x, x)


if hasattr(nn, 'Identity'):
    Identity = nn.Identity
else:
    class Identity(nn.Module):
        def execute(self, x):
            return x


class DropPath(nn.Module):
    # timm DropPath semantics (stochastic depth, per-sample). MobileSAM's
    # vit_t uses drop_path_rate=0.0, so this is effectively Identity; the
    # full behaviour is kept for faithfulness.

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        if not self.drop_prob or not self.is_training():
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = (jt.rand(shape) < keep_prob).float()
        return x / keep_prob * random_tensor


class Conv2d_BN(nn.Module):
    # torch version is a Sequential with named children 'c' (bias-free conv)
    # and 'bn'; identical key paths here.

    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1):
        super().__init__()
        self.c = nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        self.bn = nn.BatchNorm(b)
        # bn init values are overwritten by the loaded checkpoint; kept for
        # structural faithfulness when constructed without weights.
        self.bn.weight.assign(jt.full((b,), float(bn_weight_init)))
        self.bn.bias.assign(jt.zeros(b))

    def execute(self, x):
        return self.bn(self.c(x))


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, resolution, activation):
        super().__init__()
        img_size = to_2tuple(resolution)
        self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
        self.num_patches = self.patches_resolution[0] * \
            self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        n = embed_dim
        self.seq = nn.Sequential(
            Conv2d_BN(in_chans, n // 2, 3, 2, 1),
            activation(),
            Conv2d_BN(n // 2, n, 3, 2, 1),
        )

    def execute(self, x):
        return self.seq(x)


class MBConv(nn.Module):
    def __init__(self, in_chans, out_chans, expand_ratio,
                 activation, drop_path):
        super().__init__()
        self.in_chans = in_chans
        self.hidden_chans = int(in_chans * expand_ratio)
        self.out_chans = out_chans

        self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
        self.act1 = activation()

        self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans,
                               ks=3, stride=1, pad=1, groups=self.hidden_chans)
        self.act2 = activation()

        self.conv3 = Conv2d_BN(
            self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
        self.act3 = activation()

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else Identity()

    def execute(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.act2(x)

        x = self.conv3(x)

        x = self.drop_path(x)

        x = x + shortcut  # out-of-place (torch uses +=)
        x = self.act3(x)

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, out_dim, activation):
        super().__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim
        self.act = activation()
        self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
        stride_c = 2
        if (out_dim == 320 or out_dim == 448 or out_dim == 576):
            stride_c = 1
        self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
        self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

    def execute(self, x):
        if x.ndim == 3:
            H, W = self.input_resolution
            B = len(x)
            # (B, C, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        x = self.conv1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.conv3(x)
        x = x.flatten(2).permute(0, 2, 1)  # torch: flatten(2).transpose(1, 2)
        return x


class ConvLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth,
                 activation,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 out_dim=None,
                 conv_expand_ratio=4.,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint  # kept, ignored (inference-only)

        # build blocks
        self.blocks = nn.ModuleList([
            MBConv(dim, dim, conv_expand_ratio, activation,
                   drop_path[i] if isinstance(drop_path, list) else drop_path,
                   )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def execute(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.norm(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=(14, 14),
                 ):
        super().__init__()
        # (h, w)
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(
            range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        # trainable parameter, present in the checkpoint
        self.attention_biases = jt.zeros((num_heads, len(attention_offsets)))
        # torch: persistent=False buffer (absent from checkpoint) -> plain
        # numpy attribute + private flat jt index cache (underscore name is
        # never matched by load_parameters)
        self.attention_bias_idxs = np.array(idxs, dtype=np.int64).reshape(N, N)
        self._bias_idx_flat = jt.array(
            self.attention_bias_idxs.reshape(-1).astype(np.int32)).stop_grad()
        self._N = N

    def execute(self, x):  # x (B,N,C)
        B, N, _ = x.shape

        # Normalization
        x = self.norm(x)

        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.reshape(B, N, self.num_heads, -1).split(
            [self.key_dim, self.key_dim, self.d], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # gather (num_heads, N*N) -> (num_heads, N, N); identical to torch's
        # attention_biases[:, attention_bias_idxs] in both train and eval
        bias = self.attention_biases[:, self._bias_idx_flat].reshape(
            self.num_heads, self._N, self._N)

        attn = (q @ k.permute(0, 1, 3, 2)) * self.scale + bias
        attn = nn.softmax(attn, dim=-1)
        x = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


class TinyViTBlock(nn.Module):
    r"""TinyViT Block. See upstream docstring for args."""

    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 local_conv_size=3,
                 activation=nn.GELU,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else Identity()

        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads,
                              attn_ratio=1, resolution=window_resolution)

        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_activation = activation
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=mlp_activation, drop=drop)

        pad = local_conv_size // 2
        self.local_conv = Conv2d_BN(
            dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def execute(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        res_x = x
        if H == self.window_size and W == self.window_size:
            x = self.attn(x)
        else:
            x = x.reshape(B, H, W, C)
            pad_b = (self.window_size - H %
                     self.window_size) % self.window_size
            pad_r = (self.window_size - W %
                     self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0

            if padding:
                # torch: F.pad(x, (0, 0, 0, pad_r, 0, pad_b)) on (B,H,W,C)
                x = nn.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            # window partition
            # torch: view(B,nH,ws,nW,ws,C).transpose(2,3).reshape(...)
            x = x.reshape(B, nH, self.window_size, nW, self.window_size, C
                          ).permute(0, 1, 3, 2, 4, 5).reshape(
                B * nH * nW, self.window_size * self.window_size, C)
            x = self.attn(x)
            # window reverse
            x = x.reshape(B, nH, nW, self.window_size, self.window_size, C
                          ).permute(0, 1, 3, 2, 4, 5).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W]

            x = x.reshape(B, L, C)

        x = res_x + self.drop_path(x)

        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.local_conv(x)
        x = x.reshape(B, C, L).permute(0, 2, 1)

        x = x + self.drop_path(self.mlp(x))
        return x


class BasicLayer(nn.Module):
    """A basic TinyViT layer for one stage. See upstream docstring."""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0.,
                 drop_path=0., downsample=None, use_checkpoint=False,
                 local_conv_size=3,
                 activation=nn.GELU,
                 out_dim=None,
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint  # kept, ignored (inference-only)

        # build blocks
        self.blocks = nn.ModuleList([
            TinyViTBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         drop=drop,
                         drop_path=drop_path[i] if isinstance(
                             drop_path, list) else drop_path,
                         local_conv_size=local_conv_size,
                         activation=activation,
                         )
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, out_dim=out_dim, activation=activation)
        else:
            self.downsample = None

    def execute(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# upstream re-defines LayerNorm2d in tiny_vit_sam.py; reuse the common one
from .common import LayerNorm2d  # noqa: E402


class TinyViT(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 embed_dims=[96, 192, 384, 768], depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_sizes=[7, 7, 14, 7],
                 mlp_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.1,
                 use_checkpoint=False,
                 mbconv_expand_ratio=4.0,
                 local_conv_size=3,
                 layer_lr_decay=1.0,
                 ):
        super().__init__()
        self.img_size = img_size
        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio

        activation = nn.GELU

        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      embed_dim=embed_dims[0],
                                      resolution=img_size,
                                      activation=activation)

        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth decay rule (0.0 for MobileSAM vit_t)
        dpr = list(np.linspace(0, drop_path_rate, sum(depths)))

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            kwargs = dict(dim=embed_dims[i_layer],
                          input_resolution=(patches_resolution[0] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                                            patches_resolution[1] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer))),
                          depth=depths[i_layer],
                          drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                          downsample=PatchMerging if (
                              i_layer < self.num_layers - 1) else None,
                          use_checkpoint=use_checkpoint,
                          out_dim=embed_dims[min(
                              i_layer + 1, len(embed_dims) - 1)],
                          activation=activation,
                          )
            if i_layer == 0:
                layer = ConvLayer(
                    conv_expand_ratio=mbconv_expand_ratio,
                    **kwargs,
                )
            else:
                layer = BasicLayer(
                    num_heads=num_heads[i_layer],
                    window_size=window_sizes[i_layer],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    **kwargs)
            self.layers.append(layer)

        # Classifier head — unused by forward, kept for state_dict parity
        self.norm_head = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(
            embed_dims[-1], num_classes) if num_classes > 0 else Identity()

        # torch: _init_weights + set_layer_lr_decay omitted (weights are
        # loaded from the converted checkpoint; lr_scale is a training-only
        # concern)
        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dims[-1],
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )

    def forward_features(self, x):
        # x: (N, C, H, W)
        x = self.patch_embed(x)

        x = self.layers[0](x)
        start_i = 1

        for i in range(start_i, len(self.layers)):
            layer = self.layers[i]
            x = layer(x)
        B, _, C = x.shape
        x = x.reshape(B, 64, 64, C)
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)
        return x

    def execute(self, x):
        x = self.forward_features(x)
        return x
