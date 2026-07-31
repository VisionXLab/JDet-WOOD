# Ported from MobileSAM `build_sam.py`.
# for Point2RBox-v3-jittor. Inference-only.
#
# Only vit_t (MobileSAM = TinyViT encoder) is supported: the heavy
# ImageEncoderViT used by vit_h/l/b is intentionally NOT ported.
# The checkpoint argument takes the numpy-pickle produced by
# tools/convert_torch_weights.py (e.g. weights/mobile_sam.pkl),
# NOT the original .pt file.

import pickle

import jittor as jt

from .tiny_vit import TinyViT
from .prompt_encoder import PromptEncoder
from .mask_decoder import MaskDecoder
from .transformer import TwoWayTransformer
from .sam import Sam

# torch checkpoint keys that have no Jittor counterpart
_SKIP_KEY_SUFFIXES = ('num_batches_tracked', 'attention_bias_idxs')


def load_converted_checkpoint(model, pkl_path):
    with open(pkl_path, 'rb') as f:
        params = pickle.load(f)
    params = {k: jt.array(v) for k, v in params.items()
              if not k.endswith(_SKIP_KEY_SUFFIXES)}

    # Strict coverage check: a silently-skipped key (name mismatch) would
    # leave random init in place and corrupt masks without any error.
    # positional_encoding_gaussian_matrix is the known worst case
    # (random placeholder at construction, see PORTING_NOTES.md).
    # Constructed constants, not in the torch checkpoint
    # (torch registers pixel_mean/std with persistent=False).
    _NOT_IN_CKPT = ('pixel_mean', 'pixel_std')
    model_keys = {k for k, _ in model.named_parameters()
                  if not k.endswith(_NOT_IN_CKPT)}
    ckpt_keys = set(params)
    missing = model_keys - ckpt_keys
    unexpected = ckpt_keys - model_keys
    assert 'prompt_encoder.pe_layer.positional_encoding_gaussian_matrix' \
        in ckpt_keys, 'positional_encoding_gaussian_matrix missing from pkl'
    if missing:
        raise RuntimeError(f'keys missing from pkl: {sorted(missing)}')
    if unexpected:
        # Non-fatal: whether Jittor named_parameters() covers stop_grad
        # buffers is env-dependent; verify during the parity test.
        print(f'[build_sam] WARNING unexpected pkl keys: {sorted(unexpected)}')

    model.load_parameters(params)
    return model


def build_sam_vit_t(checkpoint=None):
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    mobile_sam = Sam(
        image_encoder=TinyViT(
            img_size=1024, in_chans=3, num_classes=1000,
            embed_dims=[64, 128, 160, 320],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 5, 10],
            window_sizes=[7, 7, 14, 7],
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.0,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            layer_lr_decay=0.8,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    mobile_sam.eval()
    if checkpoint is not None:
        load_converted_checkpoint(mobile_sam, checkpoint)
    return mobile_sam


sam_model_registry = {
    "vit_t": build_sam_vit_t,
}
