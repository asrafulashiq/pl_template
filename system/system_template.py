import torch
import torch.nn as nn
import abc
import hydra
from pytorch_lightning import LightningModule


# ---------------------------------------------------------------------------- #
#                               lightning module                               #
# ---------------------------------------------------------------------------- #
class LightningSystem(LightningModule):
    """ Abstract class """

    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)

    # ----------------------------------- model ---------------------------------- #
    def _create_model(self):
        """ create encoder and classifier head """

    # ----------------------------------- init ----------------------------------- #
    def setup(self, stage):
        if stage == 'fit' and self.trainer.world_size > 1:
            num_proc = self.trainer.num_nodes * self.trainer.num_processes
            self.hparams.optimizer.lr *= num_proc

    # --------------------------------- training --------------------------------- #
    @abc.abstractmethod
    def forward(self, x):
        return

    @abc.abstractmethod
    def training_step(self, batch, batch_idx):
        """ implement your own training step """

    # ----------------------------------- eval ----------------------------------- #
    def validation_step(self, batch, batch_idx, dataset_idx=0):
        pass

    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)

    # --------------------------------- optimizer -------------------------------- #
    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(
            self.hparams.optimizer,
            filter(lambda p: p.requires_grad, self.parameters()))

        if 'scheduler' in self.hparams and self.hparams.scheduler:
            scheduler = hydra.utils.instantiate(self.hparams.scheduler,
                                                optimizer=optimizer)
            return [optimizer], [scheduler]

        return optimizer

    # ----------------------------------- other ---------------------------------- #
