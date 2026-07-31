# Ported from MobileSAM `predictor.py` and
# mobile_sam/utils/transforms.py for Point2RBox-v3-jittor. Inference-only.
#
# torchvision is NOT used. torch's apply_image path is
# `resize(to_pil_image(image), target_size)` == PIL bilinear resize on uint8
# HWC — reproduced here with PIL directly, which is bit-identical to the
# torch reference. The tensor path (apply_image_torch, antialias=True) is
# not used by Point2RBox-v3 and is omitted; see PORTING_NOTES.md.

from copy import deepcopy

import numpy as np
import jittor as jt

from .sam import Sam  # noqa: F401  (type reference / import parity)


class ResizeLongestSide:
    """
    Resizes images to the longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes (numpy only; the torch-tensor
    variants of the upstream class are unused by Point2RBox-v3).
    """

    def __init__(self, target_length):
        self.target_length = target_length

    def apply_image(self, image):
        """Expects a numpy array with shape HxWxC in uint8 format."""
        from PIL import Image
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.target_length)
        # torchvision resize(PIL, (h, w), BILINEAR) == PIL resize((w, h))
        pil = Image.fromarray(image)
        resized = pil.resize((target_size[1], target_size[0]),
                             resample=Image.BILINEAR)
        return np.array(resized)

    def apply_coords(self, coords, original_size):
        """
        Expects a numpy array of length 2 in the final dimension. Requires
        the original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length)
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size):
        """Expects a numpy array shape Bx4 (XYXY)."""
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh, oldw, long_side_length):
        """Compute the output size given input size and target long side."""
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class SamPredictor:
    def __init__(self, sam_model):
        """
        Uses SAM to calculate the image embedding for an image, then allows
        repeated, efficient mask prediction given prompts.
        """
        super().__init__()
        self.model = sam_model
        self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.reset_image()

    def set_image(self, image, image_format="RGB"):
        """
        Calculates the image embeddings for the provided image (HWC uint8).
        """
        assert image_format in [
            "RGB",
            "BGR",
        ], f"image_format must be in ['RGB', 'BGR'], is {image_format}."
        if image_format != self.model.image_format:
            image = image[..., ::-1]

        # Transform the image to the form expected by the model
        input_image = self.transform.apply_image(image)
        input_image_jt = jt.array(np.ascontiguousarray(input_image)).float()
        input_image_jt = input_image_jt.permute(2, 0, 1).unsqueeze(0)

        self.set_jittor_image(input_image_jt, image.shape[:2])

    def set_jittor_image(self, transformed_image, original_image_size):
        """
        Same contract as torch's set_torch_image: input is 1x3xHxW, already
        resized so the long side equals the encoder img_size.
        """
        assert (
            len(transformed_image.shape) == 4
            and transformed_image.shape[1] == 3
            and max(*transformed_image.shape[2:]) == self.model.image_encoder.img_size
        ), (f"set_jittor_image input must be BCHW with long side "
            f"{self.model.image_encoder.img_size}.")
        self.reset_image()

        self.original_size = original_image_size
        self.input_size = tuple(transformed_image.shape[-2:])
        with jt.no_grad():
            input_image = self.model.preprocess(transformed_image)
            self.features = self.model.image_encoder(input_image)
        self.is_image_set = True

    # torch-API alias so ported call-sites keep working
    set_torch_image = set_jittor_image

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
    ):
        """
        Predict masks for the given input prompts (numpy in / numpy out).
        See upstream docstring for shapes.
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask "
                "prediction.")

        # Transform input prompts
        coords_jt, labels_jt, box_jt, mask_input_jt = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(
                point_coords, self.original_size)
            coords_jt = jt.array(np.asarray(point_coords, dtype=np.float32))
            labels_jt = jt.array(np.asarray(point_labels, dtype=np.int32))
            coords_jt = coords_jt.unsqueeze(0)
            labels_jt = labels_jt.unsqueeze(0)
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_jt = jt.array(np.asarray(box, dtype=np.float32))
            box_jt = box_jt.unsqueeze(0)
        if mask_input is not None:
            mask_input_jt = jt.array(
                np.asarray(mask_input, dtype=np.float32))
            mask_input_jt = mask_input_jt.unsqueeze(0)

        masks, iou_predictions, low_res_masks = self.predict_jittor(
            coords_jt,
            labels_jt,
            box_jt,
            mask_input_jt,
            multimask_output,
            return_logits=return_logits,
        )

        masks_np = masks[0].numpy()
        iou_predictions_np = iou_predictions[0].numpy()
        low_res_masks_np = low_res_masks[0].numpy()
        return masks_np, iou_predictions_np, low_res_masks_np

    def predict_jittor(
        self,
        point_coords,
        point_labels,
        boxes=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
    ):
        """
        Predict masks from batched jt.Var prompts already transformed to the
        input frame. See upstream predict_torch docstring for shapes.
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) before mask "
                "prediction.")

        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        with jt.no_grad():
            # Embed prompts
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points,
                boxes=boxes,
                masks=mask_input,
            )

            # Predict masks
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=self.features,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )

            # Upscale the masks to the original image resolution
            masks = self.model.postprocess_masks(
                low_res_masks, self.input_size, self.original_size)

            if not return_logits:
                masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    # torch-API alias
    predict_torch = predict_jittor

    def get_image_embedding(self):
        """Returns the image embeddings for the currently set image."""
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an "
                "embedding.")
        assert self.features is not None, \
            "Features must exist if an image has been set."
        return self.features

    def reset_image(self):
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
