from .smooth_l1_loss import SmoothL1Loss
from .smooth_focal_loss import SmoothFocalLoss
from .focal_loss import FocalLoss
from .cross_entropy_loss import CrossEntropyLoss
from .l1_loss import L1Loss
from .poly_iou_loss import PolyIoULoss
from .kf_iou_loss import KFLoss
from .gaussian_dist_loss import GDLoss
from .gaussian_dist_loss_v1 import GDLoss_v1
from .iou_loss import IoULoss
from .h2rbox_loss import H2RBoxLoss
from .kd_loss import KnowledgeDistillationKLDivLoss
from .kd_loss import IMLoss
from .rsdet_loss import RSDetLoss
from .ridet_loss import RIDetLoss
from .convex_giou_loss import ConvexGIoULoss
from .point2rbox_v2_loss import (GaussianOverlapLoss, VoronoiWatershedLoss, EdgeLoss, MMDetFocalLoss, MMDetCrossEntropyLoss,
                                 Point2RBoxV2ConsistencyLoss, gwd_sigma_loss)
from .rotated_iou_loss import RotatedIoULoss
