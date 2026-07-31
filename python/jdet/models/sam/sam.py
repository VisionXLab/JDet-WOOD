# Ported from MobileSAM `modeling/sam.py`.
# for Point2RBox-v3-jittor. Inference-only (vit_t / TinyViT encoder only;
# ImageEncoderViT for vit_h/l/b is NOT ported).
#
# Faithfulness notes:
#   - pixel_mean / pixel_std are persistent=False buffers in torch (absent
#     from the checkpoint) -> plain stop_grad attributes here.
#
# UNVERIFIED-API (check once env is ready):
#   - nn.pad(x, (0, padw, 0, padh)) pads (W_right, H_bottom) like torch F.pad
#   - nn.interpolate(..., mode='bilinear', align_corners=False)

import jittor as jt
from jittor import nn


class Sam(nn.Module):
    mask_threshold = 0.0
    image_format = "RGB"

    def __init__(
        self,
        image_encoder,
        prompt_encoder,
        mask_decoder,
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    ):
        """
        SAM predicts object masks from an image and input prompts.
        See upstream docstring for args.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        # torch: register_buffer(..., persistent=False) — not in checkpoint
        self.pixel_mean = jt.array(pixel_mean).reshape(-1, 1, 1).stop_grad()
        self.pixel_std = jt.array(pixel_std).reshape(-1, 1, 1).stop_grad()

    def execute(self, batched_input, multimask_output):
        """
        Predicts masks end-to-end from provided images and prompts.
        If prompts are not known in advance, using SamPredictor is
        recommended over calling the model directly.
        See upstream docstring for the batched_input / output dict formats.
        """
        with jt.no_grad():
            input_images = jt.stack(
                [self.preprocess(x["image"]) for x in batched_input], dim=0)
            image_embeddings = self.image_encoder(input_images)

            outputs = []
            for image_record, curr_embedding in zip(batched_input,
                                                    image_embeddings):
                if "point_coords" in image_record:
                    points = (image_record["point_coords"],
                              image_record["point_labels"])
                else:
                    points = None
                sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=points,
                    boxes=image_record.get("boxes", None),
                    masks=image_record.get("mask_inputs", None),
                )
                low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=curr_embedding.unsqueeze(0),
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                masks = self.postprocess_masks(
                    low_res_masks,
                    input_size=image_record["image"].shape[-2:],
                    original_size=image_record["original_size"],
                )
                masks = masks > self.mask_threshold
                outputs.append(
                    {
                        "masks": masks,
                        "iou_predictions": iou_predictions,
                        "low_res_logits": low_res_masks,
                    }
                )
            return outputs

    def postprocess_masks(self, masks, input_size, original_size):
        """
        Remove padding and upscale masks to the original image size.
        See upstream docstring for shapes.
        """
        masks = nn.interpolate(
            masks,
            size=(self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = nn.interpolate(
            masks, size=tuple(original_size), mode="bilinear",
            align_corners=False)
        return masks

    def preprocess(self, x):
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = nn.pad(x, (0, padw, 0, padh))
        return x
