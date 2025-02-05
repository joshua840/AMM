import torch
import torch.nn.functional as F

from .models import load_model
from .datamodule import DataModule
import torchmetrics

import typing


class ERM(DataModule):
    """
    `lightning.pytorch.Trainer` will call the functions by the following order

    for epoch in epochs:
        for batch in data:
            on_train_batch_start()
            for opt in optimizers:
                loss = train_step(batch, batch_idx, optimizer_idx)
                opt.zero_grad()
                loss.backward()
                opt.step()

        for lr_scheduler in lr_schedulers:
            lr_scheduler.step()
    """

    def __init__(
        self,
        model,
        activation_fn: str = "softplus",
        softplus_beta: float = 10,
        learning_rate: float = 1e-5,
        max_epochs: int = 10,
        freeze_bn: bool = True,
        **kwargs,
    ):
        """
        Args:
            model: model to be used
            activation_fn: activation functions of model
            softplus_beta: beta of softplus
            learning_rate: learning rate of optimizer
            max_epochs: lr schedular (cosineannealing)
            weight_decay: weight decay of optimizer
            optimizer: optimizer to be used
        """
        super().__init__(**kwargs)
        kwargs["module_name"] = "ERM"
        self.save_hyperparameters()

        self.model = load_model(
            model=self.hparams.model,
            activation_fn=self.hparams.activation_fn,
            softplus_beta=self.hparams.softplus_beta,
        )

        self.accuracy = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=1000
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.model.eval()
        return self.default_step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        self.default_step(batch, batch_idx, mode="valid")

    def test_step(self, batch, batch_idx):
        self.model.eval()
        self.default_step(batch, batch_idx, mode="test")

    def default_step(self, batch, batch_idx, mode):
        x, y = batch

        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, reduction="mean")

        acc = self.accuracy(y_hat, y)
        self.log_dict(
            {f"{mode}_loss": loss, f"{mode}_acc": acc},
            prog_bar=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if mode == "train":
            return loss

        return

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer: typing.Union[
            torch.optim.Optimizer, typing.Dict[str, torch.optim.Optimizer]
        ],
        optimizer_closure,
    ):
        # Please refer the following links for more information about optimizer_step
        # https://pytorch-lightning.readthedocs.io/en/stable/common/optimization.html

        # warm start
        if self.trainer.global_step < 500:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.learning_rate

        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def on_train_epoch_end(self):
        if self.automatic_optimization == False:
            try:
                for sch in self.lr_schedulers():
                    sch.step()
            except:
                sch = self.lr_schedulers()
                sch.step()

    def configure_optimizers(self):

        optim = torch.optim.AdamW(
            lr=self.hparams.learning_rate, params=self.model.parameters()
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optim, T_max=self.hparams.max_epochs
            ),
            "name": "lr_history",
        }

        return [optim], [scheduler]
