"""
FlashDet Detection Head.
Matches official FlashDet implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict

from ..assignment import DynamicSoftLabelAssigner
from ...utils.box_utils import distance2bbox, bbox2distance, multiclass_nms
from ...losses import QualityFocalLoss, DistributionFocalLoss, GIoULoss
from ...losses.chunked_loss import chunked_quality_focal_loss, chunked_distribution_focal_loss


class DepthwiseConvModule(nn.Module):
    """Depthwise separable conv block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 1,
        padding: int = None,
        activation: str = "LeakyReLU",
        **kwargs
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding,
            groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if activation == "LeakyReLU":
            self.act = nn.LeakyReLU(0.1, inplace=True)
        else:
            self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.act(self.bn1(self.depthwise(x)))
        x = self.act(self.bn2(self.pointwise(x)))
        return x


class Integral(nn.Module):
    """
    Distribution to integral for bounding box regression.
    Converts distribution predictions to distances.
    """
    
    def __init__(self, reg_max: int = 7):
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("project", torch.linspace(0, reg_max, reg_max + 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert distribution to integral."""
        shape = x.shape
        x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        x = F.linear(x, self.project.type_as(x))
        x = x.reshape(*shape[:-1], 4)
        return x


class FlashDetHead(nn.Module):
    """
    FlashDet detection head.
    
    Features:
    - Quality Focal Loss for classification
    - Distribution Focal Loss for regression
    - GIoU Loss for bounding box
    - Dynamic Soft Label Assignment
    
    Args:
        num_classes: Number of detection classes.
        input_channel: Input feature channel.
        feat_channels: Feature channels in head.
        stacked_convs: Number of stacked conv layers.
        kernel_size: Convolution kernel size.
        strides: Feature map strides.
        reg_max: Max value for distribution focal loss.
        activation: Activation function name.
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        input_channel: int = 96,
        feat_channels: int = 96,
        stacked_convs: int = 2,
        kernel_size: int = 5,
        strides: List[int] = None,
        reg_max: int = 7,
        activation: str = "LeakyReLU",
        loss_cfg: dict = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.strides = strides or [8, 16, 32, 64]
        self.reg_max = reg_max
        self.feat_channels = feat_channels
        self.use_sigmoid = True
        
        # Loss functions
        loss_cfg = loss_cfg or {}
        self.loss_qfl = QualityFocalLoss(
            beta=2.0, loss_weight=loss_cfg.get("qfl_weight", 1.0)
        )
        self.loss_dfl = DistributionFocalLoss(
            loss_weight=loss_cfg.get("dfl_weight", 0.25)
        )
        self.loss_bbox = GIoULoss(
            loss_weight=loss_cfg.get("bbox_weight", 2.0)
        )
        
        # Target assigner
        self.assigner = DynamicSoftLabelAssigner(topk=13, iou_factor=3.0)

        # torchtune-style chunked loss (set via enable_chunked_loss())
        self.use_chunked_loss = False
        self.chunk_size = 1024
        
        # Distribution project
        self.distribution_project = Integral(reg_max)
        
        # Build head layers
        self._init_layers(input_channel, feat_channels, stacked_convs, kernel_size, activation)
        self._init_weights()
    
    def _init_layers(self, input_channel, feat_channels, stacked_convs, kernel_size, activation):
        """Initialize head layers.
        
        Official FlashDet uses SEPARATE conv layers for each scale (not shared).
        This is the _buid_not_shared_head pattern from the official implementation.
        """
        # Separate conv layers for each scale (matches official FlashDet)
        self.cls_convs = nn.ModuleList()
        for _ in self.strides:
            convs = nn.ModuleList()
            for i in range(stacked_convs):
                in_ch = input_channel if i == 0 else feat_channels
                convs.append(
                    DepthwiseConvModule(
                        in_ch, feat_channels, kernel_size, activation=activation
                    )
                )
            self.cls_convs.append(convs)
        
        # Separate output layer for each scale (classification + regression)
        self.gfl_cls = nn.ModuleList([
            nn.Conv2d(feat_channels, self.num_classes + 4 * (self.reg_max + 1), 1)
            for _ in self.strides
        ])
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Init cls head with confidence = 0.01
        bias_cls = -4.595  # -log((1-0.01)/0.01)
        for gfl_cls in self.gfl_cls:
            nn.init.normal_(gfl_cls.weight, std=0.01)
            nn.init.constant_(gfl_cls.bias, bias_cls)
    
    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            feats: Multi-scale feature maps from FPN.
            
        Returns:
            Predictions tensor [B, num_points, num_classes + 4*(reg_max+1)].
        """
        outputs = []
        for feat, cls_convs, gfl_cls in zip(feats, self.cls_convs, self.gfl_cls):
            # Apply scale-specific conv layers
            for conv in cls_convs:
                feat = conv(feat)
            # Apply scale-specific output layer
            output = gfl_cls(feat)
            outputs.append(output.flatten(start_dim=2))
        
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)
        return outputs
    
    def get_single_level_center_priors(
        self,
        batch_size: int,
        featmap_size: Tuple[int, int],
        stride: int,
        dtype: torch.dtype,
        device: torch.device
    ) -> torch.Tensor:
        """Generate center priors for a single feature level.
        
        Official FlashDet uses top-left corner of each grid cell (no +0.5 offset).
        For a grid cell at index (i, j), the point is at:
            x = j * stride
            y = i * stride
        """
        h, w = featmap_size
        # Official FlashDet: no +0.5 offset (top-left corner, not center)
        x_range = torch.arange(w, dtype=dtype, device=device) * stride
        y_range = torch.arange(h, dtype=dtype, device=device) * stride
        y, x = torch.meshgrid(y_range, x_range, indexing="ij")
        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0],), stride)
        priors = torch.stack([x, y, strides, strides], dim=-1)
        return priors.unsqueeze(0).repeat(batch_size, 1, 1)
    
    def loss(
        self,
        preds: torch.Tensor,
        gt_meta: Dict,
        aux_preds: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute losses.

        Matches official FlashDet behaviour:
          - When aux_preds is provided, use the aux predictions to drive the
            target assignment (AGM — Assign Guidance Module).
          - Compute loss for BOTH the main head and the aux head, then add them.

        Args:
            preds: Main head prediction tensor [B, num_points, C].
            gt_meta: Ground truth metadata with 'gt_bboxes', 'gt_labels', 'img'.
            aux_preds: Auxiliary head predictions [B, num_points, C] (optional).

        Returns:
            Tuple of (total_loss, loss_states_dict).
        """
        device = preds.device
        batch_size = preds.shape[0]
        gt_bboxes_list = gt_meta["gt_bboxes"]
        gt_labels_list = gt_meta["gt_labels"]
        input_height, input_width = gt_meta["img"].shape[2:]

        # Generate center priors
        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width / stride))
            for stride in self.strides
        ]
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size, featmap_sizes[i], stride,
                dtype=torch.float32, device=device
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)

        # Split main predictions
        cls_preds, reg_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        decoded_bboxes = distance2bbox(center_priors[..., :2], dis_preds)

        if aux_preds is not None:
            # --- Official AGM: use aux predictions to guide assignment ---
            aux_cls_preds, aux_reg_preds = aux_preds.split(
                [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
            )
            aux_dis_preds = (
                self.distribution_project(aux_reg_preds) * center_priors[..., 2, None]
            )
            aux_decoded_bboxes = distance2bbox(center_priors[..., :2], aux_dis_preds)

            # Assignment driven by aux scores (detached)
            batch_assign_res = []
            for i in range(batch_size):
                gt_bboxes = torch.from_numpy(gt_bboxes_list[i]).to(device).float()
                gt_labels = torch.from_numpy(gt_labels_list[i]).to(device).long()
                assign_result = self.assigner.assign(
                    aux_cls_preds[i].detach(),
                    center_priors[i],
                    aux_decoded_bboxes[i].detach(),
                    gt_bboxes,
                    gt_labels,
                )
                batch_assign_res.append(assign_result)

            # Main head loss
            loss, loss_states = self._compute_loss(
                cls_preds, reg_preds, decoded_bboxes,
                center_priors, batch_assign_res, gt_bboxes_list
            )

            # Aux head loss (same assignment targets)
            aux_loss, aux_loss_states = self._compute_loss(
                aux_cls_preds, aux_reg_preds, aux_decoded_bboxes,
                center_priors, batch_assign_res, gt_bboxes_list
            )
            loss = loss + aux_loss
            for k, v in aux_loss_states.items():
                loss_states[f"aux_{k}"] = v
        else:
            # No aux head — assign using main predictions
            batch_assign_res = []
            for i in range(batch_size):
                gt_bboxes = torch.from_numpy(gt_bboxes_list[i]).to(device).float()
                gt_labels = torch.from_numpy(gt_labels_list[i]).to(device).long()
                assign_result = self.assigner.assign(
                    cls_preds[i].detach(),
                    center_priors[i],
                    decoded_bboxes[i].detach(),
                    gt_bboxes,
                    gt_labels,
                )
                batch_assign_res.append(assign_result)

            loss, loss_states = self._compute_loss(
                cls_preds, reg_preds, decoded_bboxes,
                center_priors, batch_assign_res, gt_bboxes_list
            )

        return loss, loss_states
    
    def _compute_loss(
        self,
        cls_preds: torch.Tensor,
        reg_preds: torch.Tensor,
        decoded_bboxes: torch.Tensor,
        center_priors: torch.Tensor,
        assign_results: List,
        gt_bboxes_list: List
    ) -> Tuple[torch.Tensor, Dict]:
        """Compute losses from assignment results."""
        device = cls_preds.device

        # Gather targets
        all_labels = []
        all_label_scores = []
        all_label_weights = []
        all_bbox_targets = []
        all_dist_targets = []
        num_pos_list = []

        for i, assign_result in enumerate(assign_results):
            num_priors = center_priors.shape[1]
            gt_bboxes = torch.from_numpy(gt_bboxes_list[i]).to(device).float()

            labels = center_priors.new_full((num_priors,), self.num_classes, dtype=torch.long)
            label_scores = center_priors.new_zeros((num_priors,))
            # label_weights: 1.0 for positives AND negatives (matches official)
            label_weights = center_priors.new_zeros((num_priors,))
            bbox_targets = center_priors.new_zeros((num_priors, 4))
            dist_targets = center_priors.new_zeros((num_priors, 4))

            pos_inds = (assign_result.gt_inds > 0).nonzero(as_tuple=False).squeeze(-1)
            neg_inds = (assign_result.gt_inds == 0).nonzero(as_tuple=False).squeeze(-1)
            num_pos = pos_inds.numel()
            num_pos_list.append(num_pos)

            if num_pos > 0:
                pos_gt_inds = assign_result.gt_inds[pos_inds] - 1
                pos_gt_bboxes = gt_bboxes[pos_gt_inds]

                labels[pos_inds] = assign_result.labels[pos_inds]
                label_scores[pos_inds] = assign_result.max_overlaps[pos_inds]
                label_weights[pos_inds] = 1.0
                bbox_targets[pos_inds] = pos_gt_bboxes

                pos_strides = center_priors[i, pos_inds, 2:3]
                raw_distances = bbox2distance(
                    center_priors[i, pos_inds, :2], pos_gt_bboxes, max_dis=None
                )
                dist_targets[pos_inds] = (raw_distances / pos_strides).clamp(
                    min=0, max=self.reg_max - 0.01
                )

            if neg_inds.numel() > 0:
                label_weights[neg_inds] = 1.0

            all_labels.append(labels)
            all_label_scores.append(label_scores)
            all_label_weights.append(label_weights)
            all_bbox_targets.append(bbox_targets)
            all_dist_targets.append(dist_targets)

        # Stack
        labels = torch.stack(all_labels, dim=0).reshape(-1)
        label_scores = torch.stack(all_label_scores, dim=0).reshape(-1)
        label_weights = torch.stack(all_label_weights, dim=0).reshape(-1)
        bbox_targets = torch.stack(all_bbox_targets, dim=0).reshape(-1, 4)
        dist_targets = torch.stack(all_dist_targets, dim=0).reshape(-1, 4)

        num_total_pos = max(sum(num_pos_list), 1)

        # Flatten predictions
        cls_preds = cls_preds.reshape(-1, self.num_classes)
        reg_preds = reg_preds.reshape(-1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)

        # Quality Focal Loss — with per-sample label_weights
        if self.use_chunked_loss:
            loss_qfl = self.loss_qfl.loss_weight * chunked_quality_focal_loss(
                cls_preds, (labels, label_scores),
                weight=label_weights, avg_factor=num_total_pos,
                chunk_size=self.chunk_size,
            )
        else:
            loss_qfl = self.loss_qfl(
                cls_preds, (labels, label_scores),
                weight=label_weights, avg_factor=num_total_pos
            )

        # Box losses for positive samples only
        pos_inds = ((labels >= 0) & (labels < self.num_classes)).nonzero(as_tuple=False).squeeze(-1)

        if pos_inds.numel() > 0:
            pos_decoded_bboxes = decoded_bboxes[pos_inds]
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_reg_preds = reg_preds[pos_inds]
            pos_dist_targets = dist_targets[pos_inds]

            # Weight by max predicted class score (detached)
            weight_targets = cls_preds[pos_inds].detach().sigmoid().max(dim=1)[0]
            bbox_avg_factor = max(weight_targets.sum().item(), 1.0)

            # GIoU Loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bboxes, pos_bbox_targets,
                weight=weight_targets, avg_factor=bbox_avg_factor
            )

            # Distribution Focal Loss
            if self.use_chunked_loss:
                loss_dfl = self.loss_dfl.loss_weight * chunked_distribution_focal_loss(
                    pos_reg_preds.reshape(-1, self.reg_max + 1),
                    pos_dist_targets.reshape(-1),
                    weight=weight_targets.unsqueeze(-1).expand(-1, 4).reshape(-1),
                    avg_factor=4.0 * bbox_avg_factor,
                    chunk_size=self.chunk_size,
                )
            else:
                loss_dfl = self.loss_dfl(
                    pos_reg_preds.reshape(-1, self.reg_max + 1),
                    pos_dist_targets.reshape(-1),
                    weight=weight_targets.unsqueeze(-1).expand(-1, 4).reshape(-1),
                    avg_factor=4.0 * bbox_avg_factor,
                )
        else:
            loss_bbox = reg_preds.sum() * 0
            loss_dfl = reg_preds.sum() * 0
        
        loss = loss_qfl + loss_bbox + loss_dfl
        
        loss_states = {
            "loss_qfl": loss_qfl.detach(),
            "loss_bbox": loss_bbox.detach(),
            "loss_dfl": loss_dfl.detach(),
            "num_pos": torch.tensor(num_total_pos, device=device)
        }
        
        return loss, loss_states
    
    def get_bboxes(
        self,
        preds: torch.Tensor,
        img_metas: Dict,
        score_thr: float = 0.05,  # Official FlashDet uses 0.05 for NMS
        nms_thr: float = 0.6
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Decode predictions to bboxes.
        
        Args:
            preds: Predictions [B, num_points, C].
            img_metas: Image metadata.
            score_thr: Score threshold.
            nms_thr: NMS threshold.
            
        Returns:
            List of (det_bboxes, det_labels) per image.
        """
        device = preds.device
        batch_size = preds.shape[0]
        input_height, input_width = img_metas["img"].shape[2:]
        input_shape = (input_height, input_width)
        
        # Generate center priors
        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width / stride))
            for stride in self.strides
        ]
        mlvl_center_priors = [
            self.get_single_level_center_priors(
                batch_size, featmap_sizes[i], stride,
                dtype=torch.float32, device=device
            )
            for i, stride in enumerate(self.strides)
        ]
        center_priors = torch.cat(mlvl_center_priors, dim=1)
        
        # Split predictions
        cls_preds, reg_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        
        # Decode boxes
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_preds.sigmoid()
        
        # NMS per image
        result_list = []
        for i in range(batch_size):
            # Add dummy background class
            score = scores[i]
            bbox = bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            
            dets, labels = multiclass_nms(bbox, score, score_thr, nms_thr, max_num=100)
            result_list.append((dets, labels))
        
        return result_list
