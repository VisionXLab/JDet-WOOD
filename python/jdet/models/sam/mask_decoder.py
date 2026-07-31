# Ported from MobileSAM `modeling/mask_decoder.py`.
# for Point2RBox-v3-jittor. Inference-only.
#
# torch.repeat_interleave(x, n, dim=0) is emulated via
# unsqueeze+expand+reshape (row-consecutive repetition, identical layout).
#
# UNVERIFIED-API (check once env is ready):
#   - nn.ConvTranspose(in, out, kernel_size, stride) argument order/name
#   - nn.relu / nn.ReLU availability for the functional call in MLP

import jittor as jt
from jittor import nn

from .common import LayerNorm2d


def _repeat_interleave_dim0(x, n):
    # torch.repeat_interleave(x, n, dim=0): each row repeated n times
    # consecutively: (B, ...) -> (B*n, ...)
    b = x.shape[0]
    rest = list(x.shape[1:])
    x = x.unsqueeze(1).expand([b, n] + rest)
    return x.reshape([b * n] + rest)


class MaskDecoder(nn.Module):
    def __init__(
        self,
        *,
        transformer_dim,
        transformer,
        num_multimask_outputs=3,
        activation=nn.GELU,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    ):
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture. See upstream docstring for args.
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose(transformer_dim, transformer_dim // 4,
                             kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose(transformer_dim // 4, transformer_dim // 8,
                             kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)
            ]
        )

        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens,
            iou_head_depth
        )

    def execute(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
        multimask_output,
    ):
        """
        Predict masks given image and prompt embeddings.
        See upstream docstring for shapes.
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings,
        image_pe,
        sparse_prompt_embeddings,
        dense_prompt_embeddings,
    ):
        """Predicts masks. See 'execute' for more details."""
        # Concatenate output tokens
        output_tokens = jt.concat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.shape[0], -1, -1)
        tokens = jt.concat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = _repeat_interleave_dim0(image_embeddings, tokens.shape[0])
        src = src + dense_prompt_embeddings
        pos_src = _repeat_interleave_dim0(image_pe, tokens.shape[0])
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.permute(0, 2, 1).reshape(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = jt.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.reshape(b, c, h * w)
                 ).reshape(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from MaskFormer (see upstream header).
class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        sigmoid_output=False,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        # jittor nn.ModuleList rejects generators — pass a list
        self.layers = nn.ModuleList(
            [nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])]
        )
        self.sigmoid_output = sigmoid_output

    def execute(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = jt.sigmoid(x)
        return x
