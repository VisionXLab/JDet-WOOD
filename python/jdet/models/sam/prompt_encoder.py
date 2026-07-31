# Ported from MobileSAM `modeling/prompt_encoder.py`.
# for Point2RBox-v3-jittor. Inference-only.
#
# Faithfulness notes:
#   - `positional_encoding_gaussian_matrix` is a PERSISTENT buffer in torch
#     (present in the checkpoint) — here a plain jt.Var attribute with
#     stop_grad(); load_parameters must overwrite it (verify the converted
#     checkpoint value actually lands, otherwise masks will be garbage).
#   - torch's masked in-place adds in _embed_points are rewritten as
#     out-of-place mask-multiply adds (numerically identical, safer in
#     Jittor).
#   - torch.empty((bs, 0, C)) + repeated cat is rewritten as a list that is
#     concatenated once (zero-size tensors are fragile in Jittor).
#
# UNVERIFIED-API (check once env is ready):
#   - nn.Embedding(1, dim) exposes .weight of shape (1, dim)
#   - jt.cumsum(x, dim)
#   - nn.ModuleList numeric naming ("point_embeddings.0.weight")

import numpy as np

import jittor as jt
from jittor import nn

from .common import LayerNorm2d


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim,
        image_embedding_size,
        input_image_size,
        mask_in_chans,
        activation=nn.GELU,
    ):
        """
        Encodes prompts for input to SAM's mask decoder.
        See upstream docstring for args.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim)
                            for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0],
                                4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self):
        """1 x embed_dim x embedding_h x embedding_w positional encoding."""
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points, labels, pad):
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = jt.zeros((points.shape[0], 1, 2))
            padding_label = -jt.ones((labels.shape[0], 1))
            points = jt.concat([points, padding_point], dim=1)
            labels = jt.concat([labels.float(), padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(
            points, self.input_image_size)
        # torch (in-place masked ops):
        #   point_embedding[labels == -1] = 0.0
        #   point_embedding[labels == -1] += self.not_a_point_embed.weight
        #   point_embedding[labels == 0] += self.point_embeddings[0].weight
        #   point_embedding[labels == 1] += self.point_embeddings[1].weight
        # out-of-place equivalent:
        m_neg1 = (labels == -1).float().unsqueeze(-1)
        m_0 = (labels == 0).float().unsqueeze(-1)
        m_1 = (labels == 1).float().unsqueeze(-1)
        point_embedding = point_embedding * (1.0 - m_neg1)
        point_embedding = point_embedding + m_neg1 * self.not_a_point_embed.weight
        point_embedding = point_embedding + m_0 * self.point_embeddings[0].weight
        point_embedding = point_embedding + m_1 * self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes):
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(
            coords, self.input_image_size)
        # torch adds in place per corner; out-of-place equivalent:
        corner_offsets = jt.concat([
            self.point_embeddings[2].weight,   # corner 0
            self.point_embeddings[3].weight,   # corner 1
        ], dim=0).unsqueeze(0)                 # (1, 2, embed_dim)
        corner_embedding = corner_embedding + corner_offsets
        return corner_embedding

    def _embed_masks(self, masks):
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(self, points, boxes, masks):
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def execute(self, points, boxes, masks):
        """
        Embeds different types of prompts, returning sparse and dense
        embeddings. See upstream docstring for shapes.
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_list = []
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(
                coords, labels, pad=(boxes is None))
            sparse_list.append(point_embeddings)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_list.append(box_embeddings)
        if sparse_list:
            sparse_embeddings = (sparse_list[0] if len(sparse_list) == 1
                                 else jt.concat(sparse_list, dim=1))
        else:
            sparse_embeddings = jt.zeros((bs, 0, self.embed_dim))

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(
                1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0],
                self.image_embedding_size[1])

        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats=64, scale=None):
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        # torch: persistent buffer (IN the checkpoint); values here are
        # placeholders overwritten by load_parameters.
        self.positional_encoding_gaussian_matrix = (
            scale * jt.randn((2, num_pos_feats))).stop_grad()

    def _pe_encoding(self, coords):
        """Positionally encode points that are normalized to [0,1]."""
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return jt.concat([jt.sin(coords), jt.cos(coords)], dim=-1)

    def execute(self, size):
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        grid = jt.ones((h, w), dtype='float32')
        y_embed = jt.cumsum(grid, dim=0) - 0.5
        x_embed = jt.cumsum(grid, dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(jt.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(self, coords_input, image_size):
        """Positionally encode points that are not normalized to [0,1]."""
        # out-of-place (torch clones then writes columns)
        x = coords_input[..., 0] / image_size[1]
        y = coords_input[..., 1] / image_size[0]
        coords = jt.stack([x, y], dim=-1)
        return self._pe_encoding(coords.float())  # B x N x C
