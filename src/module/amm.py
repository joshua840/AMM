from .erm import ERM
import torch.nn.functional as F
import torch
from .cam import get_interpreter


class AMM(ERM):
    def __init__(
        self,
        h_method: str,
        f_loss: str,
        h_lambda: float = 1,
        h_target_layer: str = "layer4",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        Interpreter = get_interpreter(self.hparams.h_method)
        self.interpreter = Interpreter(
            model=self.model, target_layer=self.hparams.h_target_layer
        )

    def default_step(self, batch, batch_idx, mode):
        x, y = batch

        with torch.enable_grad():
            x.requires_grad = True

            y_hat = self(x)

            cam = self.interpreter.compute_cams(
                logits=y_hat[range(len(y)), y], normalized=True
            )

            if not hasattr(self, "mask"):
                self.mask = torch.zeros_like(cam[0:1]).detach()
                self.mask[:, :, 0] = 1
                self.mask[:, :, -1] = 1
                self.mask[:, 0, :] = 1
                self.mask[:, -1, :] = 1

            ce_loss = F.cross_entropy(y_hat, y, reduction="mean")
            f_loss = F.mse_loss(cam, self.mask, reduction="mean")
            loss = ce_loss + self.hparams.h_lambda * f_loss
            acc = self.accuracy(y_hat, y)

        self.log_dict(
            {
                f"{mode}_ce_loss": ce_loss,
                f"{mode}_f_loss": f_loss,
                f"{mode}_loss": loss,
                f"{mode}_acc": acc,
            },
            prog_bar=True,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if mode == "train":
            return loss

        return
