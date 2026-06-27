"""Self-Supervised Learning (SSL) Pretraining for FlashDet.

Provides contrastive pretraining for detection backbones using
a BYOL/MoCo-style approach. The pretrained backbone can then be
used to initialise a detector for fine-tuning on downstream tasks.

Usage::

    from flashdet.engine import SSLTrainer

    trainer = SSLTrainer(
        ssl_method="byol",        # "byol" | "moco" | "simclr"
        train_images="path/to/unlabeled/images",
        epochs=100,
    )
    backbone_path = ssl.pretrain()

    # Then fine-tune:
    from flashdet.engine import Trainer
    trainer = Trainer(finetune=backbone_path, ...)
    trainer.train()
"""

import os
import copy
import math
import logging
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashdet.engine.core.ema import ModelEMA
from flashdet.utils import setup_logger, AverageMeter

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """MLP projection head used in contrastive learning."""

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionHead(nn.Module):
    """MLP prediction head (BYOL-specific asymmetry)."""

    def __init__(self, in_dim: int = 256, hidden_dim: int = 1024, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SSLTrainer:
    """Self-Supervised Learning pretrainer for detection backbones.

    Trains a backbone using contrastive or self-distillation objectives
    on unlabeled images. The resulting backbone weights can be used to
    initialize any FlashDet detector for faster convergence on small datasets.

    Supported methods:

    - **BYOL** (Bootstrap Your Own Latent): online/target network with
      momentum updates; no negative pairs needed.
    - **MoCo** (Momentum Contrast): contrastive learning with a momentum
      encoder and a queue of negative keys.
    - **SimCLR** (Simple Contrastive Learning): NT-Xent contrastive loss
      with in-batch negatives.

    Args:
        ssl_method: One of ``"byol"``, ``"moco"``, or ``"simclr"``.
        backbone_name: Backbone architecture to pretrain.
        backbone_size: Size variant of the backbone.
        proj_dim: Projection head output dimension.
        queue_size: Size of the negative queue (MoCo only).
        temperature: InfoNCE temperature (MoCo/SimCLR).
        momentum: EMA momentum for target network (BYOL/MoCo).
        epochs: Number of pretraining epochs.
        batch_size: Batch size.
        lr: Base learning rate.
        workers: Dataloader workers.
        save_dir: Output directory.
        device: Training device.
        amp: Enable automatic mixed precision.
        train_images: Path to unlabeled training images.
        input_size: Input image size.
    """

    SUPPORTED_METHODS = ("byol", "moco", "simclr")

    def __init__(
        self,
        ssl_method: str = "byol",
        backbone_name: str = "LiteBackbone",
        backbone_size: str = "1.0x",
        proj_dim: int = 256,
        queue_size: int = 65536,
        temperature: float = 0.07,
        momentum: float = 0.996,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 0.03,
        workers: int = 4,
        save_dir: str = "workspace/ssl_output",
        device: str = "cuda",
        amp: bool = True,
        train_images: Optional[str] = None,
        input_size: int = 224,
    ):
        assert ssl_method in self.SUPPORTED_METHODS, (
            f"ssl_method must be one of {self.SUPPORTED_METHODS}, got '{ssl_method}'"
        )
        self.ssl_method = ssl_method
        self.backbone_name = backbone_name
        self.backbone_size = backbone_size
        self.proj_dim = proj_dim
        self.queue_size = queue_size
        self.temperature = temperature
        self.momentum = momentum
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.workers = workers
        self.save_dir = save_dir
        self.train_images = train_images
        self.input_size = input_size
        self.amp = amp

        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        os.makedirs(self.save_dir, exist_ok=True)
        self._logger = setup_logger("FlashDet-SSL", self.save_dir)

    def _build_backbone(self) -> nn.Module:
        """Build the backbone network to pretrain."""
        from flashdet.registry import BACKBONES
        name = self.backbone_name

        if name in BACKBONES:
            return BACKBONES.get(name)()

        for registered_name in BACKBONES.list():
            if registered_name.lower() == name.lower():
                return BACKBONES.get(registered_name)()

        from flashdet.models.backbone.lite_backbone import LiteBackbone
        return LiteBackbone(model_size=self.backbone_size)

    def _get_backbone_out_dim(self, backbone: nn.Module) -> int:
        """Infer the output feature dimension of the backbone."""
        backbone.eval()
        dummy = torch.zeros(1, 3, self.input_size, self.input_size)
        with torch.no_grad():
            feat = backbone(dummy)
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        return feat.shape[1]

    def _build_ssl_augmentation(self):
        """Build SSL-specific augmentation pipeline (two random crops)."""
        import torchvision.transforms as T
        return T.Compose([
            T.RandomResizedCrop(self.input_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _build_dataloader(self):
        """Build a dataloader for unlabeled images."""
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader

        transform = self._build_ssl_augmentation()

        class TwoViewTransform:
            def __init__(self, base_transform):
                self.base = base_transform
            def __call__(self, img):
                return self.base(img), self.base(img)

        dataset = ImageFolder(self.train_images, transform=TwoViewTransform(transform))
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.workers, pin_memory=True, drop_last=True,
        )

    def _byol_loss(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Negative cosine similarity (BYOL loss)."""
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return 2 - 2 * (p * z).sum(dim=-1).mean()

    def _infonce_loss(self, q: torch.Tensor, k: torch.Tensor, queue: torch.Tensor = None) -> torch.Tensor:
        """InfoNCE contrastive loss (MoCo/SimCLR)."""
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        l_pos = torch.einsum("nc,nc->n", q, k).unsqueeze(-1)

        if queue is not None:
            l_neg = torch.einsum("nc,ck->nk", q, queue.clone().detach())
            logits = torch.cat([l_pos, l_neg], dim=1) / self.temperature
        else:
            sim_matrix = torch.einsum("nc,mc->nm", q, k) / self.temperature
            B = q.shape[0]
            mask = ~torch.eye(B, dtype=torch.bool, device=q.device)
            logits = torch.cat([l_pos / self.temperature, sim_matrix.masked_select(mask).view(B, -1)], dim=1)

        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)

    def pretrain(self) -> str:
        """Run the SSL pretraining loop.

        Returns:
            Path to the saved pretrained backbone weights.
        """
        self._logger.info("=" * 60)
        self._logger.info(f"SSL Pretraining ({self.ssl_method.upper()})")
        self._logger.info("=" * 60)

        backbone = self._build_backbone().to(self.device)
        feat_dim = self._get_backbone_out_dim(backbone.cpu())
        backbone = backbone.to(self.device)

        projector = ProjectionHead(feat_dim, out_dim=self.proj_dim).to(self.device)

        class _BackboneWrapper(nn.Module):
            """Wraps a backbone that returns a list of feature maps,
            extracting only the last stage for contrastive learning."""
            def __init__(self, bb):
                super().__init__()
                self.bb = bb
            def forward(self, x):
                out = self.bb(x)
                return out[-1] if isinstance(out, (list, tuple)) else out

        online_net = nn.Sequential(_BackboneWrapper(backbone), nn.AdaptiveAvgPool2d(1), nn.Flatten(), projector)

        if self.ssl_method == "byol":
            predictor = PredictionHead(self.proj_dim, out_dim=self.proj_dim).to(self.device)
            target_net = copy.deepcopy(online_net)
            for p in target_net.parameters():
                p.requires_grad = False
        else:
            predictor = None
            target_net = copy.deepcopy(online_net)
            for p in target_net.parameters():
                p.requires_grad = False

        queue = None
        if self.ssl_method == "moco":
            queue = F.normalize(torch.randn(self.proj_dim, self.queue_size, device=self.device), dim=0)

        params = list(online_net.parameters())
        if predictor is not None:
            params += list(predictor.parameters())
        optimizer = torch.optim.SGD(params, lr=self.lr, momentum=0.9, weight_decay=1e-4)

        scaler = None
        if self.amp and self.device.type == "cuda":
            scaler = torch.amp.GradScaler("cuda")

        dataloader = self._build_dataloader()
        self._logger.info(f"Dataset: {len(dataloader.dataset)} images, {len(dataloader)} batches")

        for epoch in range(self.epochs):
            lr = self._cosine_lr(epoch)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            loss_meter = AverageMeter("SSL Loss")
            online_net.train()
            if predictor is not None:
                predictor.train()

            for batch_idx, ((view1, view2), _) in enumerate(dataloader):
                view1, view2 = view1.to(self.device), view2.to(self.device)

                with torch.amp.autocast(self.device.type, enabled=scaler is not None):
                    z1 = online_net(view1)
                    z2 = online_net(view2)

                    with torch.no_grad():
                        t1 = target_net(view1)
                        t2 = target_net(view2)

                    if self.ssl_method == "byol":
                        p1 = predictor(z1)
                        p2 = predictor(z2)
                        loss = self._byol_loss(p1, t2) + self._byol_loss(p2, t1)
                    elif self.ssl_method == "moco":
                        loss = self._infonce_loss(z1, t2, queue)
                        self._dequeue_and_enqueue(queue, t2)
                    else:  # simclr
                        loss = self._infonce_loss(z1, z2)

                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                self._update_target(online_net, target_net, epoch)
                loss_meter.update(loss.item())

            self._logger.info(f"Epoch {epoch+1}/{self.epochs} | SSL Loss: {loss_meter.avg:.4f} | LR: {lr:.6f}")

        save_path = os.path.join(self.save_dir, "backbone_pretrained.pth")
        torch.save(backbone.state_dict(), save_path)
        self._logger.info(f"Pretrained backbone saved to {save_path}")
        return save_path

    def _cosine_lr(self, epoch: int) -> float:
        return self.lr * 0.5 * (1 + math.cos(math.pi * epoch / self.epochs))

    def _update_target(self, online: nn.Module, target: nn.Module, epoch: int):
        m = 1 - (1 - self.momentum) * (math.cos(math.pi * epoch / self.epochs) + 1) / 2
        for op, tp in zip(online.parameters(), target.parameters()):
            tp.data.mul_(m).add_(op.data, alpha=1 - m)

    @staticmethod
    def _dequeue_and_enqueue(queue: torch.Tensor, keys: torch.Tensor):
        keys = F.normalize(keys, dim=-1)
        batch_size = keys.shape[0]
        queue_size = queue.shape[1]
        queue[:, batch_size:] = queue[:, :-batch_size].clone()
        queue[:, :batch_size] = keys.T
