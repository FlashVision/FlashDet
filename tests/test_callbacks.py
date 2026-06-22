"""Unit tests for training callbacks."""

from flashdet.engine.core.callbacks import Callback, CallbackList, EarlyStopping


class MockTrainer:
    pass


def test_callback_fires():
    called = []

    class TestCB(Callback):
        def on_epoch_end(self, trainer, epoch, metrics):
            called.append(epoch)

    cb_list = CallbackList([TestCB()])
    cb_list.fire("on_epoch_end", MockTrainer(), 1, {"loss": 0.5})
    cb_list.fire("on_epoch_end", MockTrainer(), 2, {"loss": 0.4})

    assert called == [1, 2]


def test_early_stopping_triggers():
    es = EarlyStopping(patience=3, metric="val_mAP", mode="max")
    trainer = MockTrainer()

    es.on_epoch_end(trainer, 1, {"val_mAP": 0.5})
    assert not es.should_stop

    es.on_epoch_end(trainer, 2, {"val_mAP": 0.4})
    es.on_epoch_end(trainer, 3, {"val_mAP": 0.3})
    es.on_epoch_end(trainer, 4, {"val_mAP": 0.2})
    assert es.should_stop


def test_early_stopping_resets_on_improvement():
    es = EarlyStopping(patience=3, metric="loss", mode="min")
    trainer = MockTrainer()

    es.on_epoch_end(trainer, 1, {"loss": 1.0})
    es.on_epoch_end(trainer, 2, {"loss": 1.1})
    es.on_epoch_end(trainer, 3, {"loss": 0.9})  # improvement resets
    es.on_epoch_end(trainer, 4, {"loss": 1.0})
    es.on_epoch_end(trainer, 5, {"loss": 1.1})

    assert not es.should_stop  # only 2 waits after reset
