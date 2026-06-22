"""DETR prediction heads (class + bbox FFNs)."""

import torch
import torch.nn as nn


class DETRHead(nn.Module):
    """FFN prediction heads for DETR (classification + bounding box)."""

    def __init__(self, d_model: int = 256, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.bbox_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.class_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.bbox_head.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, hs: torch.Tensor):
        """Args: hs — decoder output [B, num_queries, d_model].
        Returns: (pred_logits, pred_boxes)."""
        pred_logits = self.class_head(hs)
        pred_boxes = self.bbox_head(hs).sigmoid()
        return pred_logits, pred_boxes
