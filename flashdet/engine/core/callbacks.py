"""Training callbacks / hooks system.

Extensible hook points for the training loop. Users can add custom
behavior without modifying the Trainer source code.

Usage:
    from flashdet.engine.core.callbacks import Callback, CallbackList

    class WandbLogger(Callback):
        def on_epoch_end(self, trainer, epoch, metrics):
            wandb.log(metrics, step=epoch)

    trainer = Trainer(...)
    trainer.add_callback(WandbLogger())
    trainer.train()
"""

from typing import Any, Dict, List, Optional


class Callback:
    """Base class for training callbacks.

    Override any method to hook into the training loop.
    All methods receive the trainer instance as the first argument.
    """

    def on_train_start(self, trainer: Any) -> None:
        """Called once before training begins."""

    def on_train_end(self, trainer: Any, metrics: Dict) -> None:
        """Called once after training completes."""

    def on_epoch_start(self, trainer: Any, epoch: int) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        """Called at the end of each epoch with computed metrics."""

    def on_batch_start(self, trainer: Any, batch_idx: int, batch: Any) -> None:
        """Called before each training batch."""

    def on_batch_end(self, trainer: Any, batch_idx: int, loss: float) -> None:
        """Called after each training batch."""

    def on_val_start(self, trainer: Any) -> None:
        """Called before validation."""

    def on_val_end(self, trainer: Any, metrics: Dict) -> None:
        """Called after validation with results."""

    def on_checkpoint(self, trainer: Any, path: str, is_best: bool) -> None:
        """Called when a checkpoint is saved."""


class CallbackList:
    """Manages a list of callbacks, dispatching events to all of them."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks: List[Callback] = callbacks or []

    def add(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def fire(self, event: str, *args, **kwargs) -> None:
        """Fire an event on all registered callbacks."""
        for cb in self.callbacks:
            method = getattr(cb, event, None)
            if method:
                method(*args, **kwargs)


class EarlyStopping(Callback):
    """Stop training when a metric stops improving."""

    def __init__(self, patience: int = 20, metric: str = "val_mAP", mode: str = "max"):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.wait = 0
        self.should_stop = False

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        value = metrics.get(self.metric)
        if value is None:
            return

        improved = (value > self.best) if self.mode == "max" else (value < self.best)
        if improved:
            self.best = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
                print(f"EarlyStopping: {self.metric} did not improve for {self.patience} epochs. Stopping.")


class LRSchedulerCallback(Callback):
    """Step a PyTorch LR scheduler at each epoch."""

    def __init__(self, scheduler):
        self.scheduler = scheduler

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        self.scheduler.step()


class CSVLogger(Callback):
    """Log training metrics to a CSV file."""

    def __init__(self, path: str = "training_log.csv"):
        self.path = path
        self._initialized = False

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        import csv

        row = {"epoch": epoch, **metrics}
        if not self._initialized:
            with open(self.path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)
            self._initialized = True
        else:
            with open(self.path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)


class TensorBoardCallback(Callback):
    """Log metrics to TensorBoard."""

    def __init__(self, log_dir: str = "runs"):
        self.log_dir = log_dir
        self._writer = None

    def on_train_start(self, trainer: Any) -> None:
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(self.log_dir)

    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict) -> None:
        if self._writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(key, value, epoch)

    def on_train_end(self, trainer: Any, metrics: Dict) -> None:
        if self._writer:
            self._writer.close()
