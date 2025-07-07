"""PyTorch Lightning implementation of the convolutional auto-encoder
===========================================================================

Usage example
-------------
model = LitConvAutoencoder(
encoder_step_size=[256,128,64],
pooling_list=[2,2,2],
decoder_step_size=[64,128,256],
upsampling_list=[2,2,2],
embedding_size=32,
conv_size=16,
learning_rate=3e-4,
coef_ln=1e-3,
coef_contrastive=1e-2,
coef_divergence=1e-2,
)
trainer = pl.Trainer(max_epochs=100, accelerator="auto")
trainer.fit(model, train_dataloader)

The helper methods `get_embedding` and `generate_spectra` reproduce the
original convenience interface.
"""
from __future__ import annotations

import math
from typing import Any, List, Sequence

import numpy as np
import pytorch_lightning as pl

import torch

from torch.utils.data import DataLoader

from m3_learning.nn.Regularization.Regularizers import ContrastiveLoss, DivergenceLoss
from . import models


class LitConvAutoencoder(pl.LightningModule):
    """LightningModule wrapping a convolutional auto-encoder.

    Parameters
    ----------
    encoder_step_size, pooling_list, decoder_step_size, upsampling_list : Sequence[int]
        Architecture description passed straight to *Encoder* / *Decoder*.
    embedding_size : int
        Size of the latent space.
    conv_size : int
        Base number of convolutional chatorch.nnels.
    learning_rate : float, default=3e-5
        Initial learning rate for *Adam*.
    coef_ln, coef_contrastive, coef_divergence : float, default=0.
        Weights for the additional regularisation terms.
    ln_parm : int, default=1
        Order of the L-n norm applied to the embedding.
    max_learning_rate : float, default=1e-4
        Peak LR for the cyclic scheduler (ignored if *use_scheduler=False*).
    use_scheduler : bool, default=True
        Whether to wrap the optimiser in a *CyclicLR* scheduler.
    """

    def __init__(
        self,
        *,
        encoder_step_size: Sequence[int],
        pooling_list: Sequence[int],
        decoder_step_size: Sequence[int],
        upsampling_list: Sequence[int],
        embedding_size: int,
        conv_size: int,
        learning_rate: float = 3e-5,
        coef_ln: float = 0.0,
        coef_contrastive: float = 0.0,
        coef_divergence: float = 0.0,
        ln_parm: int = 1,
        max_learning_rate: float = 1e-4,
        use_scheduler: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # Preserve every hyper-parameter for checkpointing & reproducibility
        self.save_hyperparameters()

        # ------------------------------------------------------------------
        # Network definition – reuse exactly the same building blocks
        # ------------------------------------------------------------------
        self.encoder: torch.nn.Module = models.Encoder(
            original_step_size=list(encoder_step_size),
            pooling_list=list(pooling_list),
            embedding_size=embedding_size,
            conv_size=conv_size,
        )

        self.decoder: torch.nn.Module = models.Decoder(
            original_step_size=list(decoder_step_size),
            upsampling_list=list(upsampling_list),
            embedding_size=embedding_size,
            conv_size=conv_size,
            pooling_list=list(pooling_list),
        )

        # Wrap inside a single *AutoEncoder* if you prefer to keep parity
        self.autoencoder: torch.nn.Module = models.AutoEncoder(self.encoder, self.decoder)

        # Extra losses
        self.contrastive_loss_fn = ContrastiveLoss(coef_contrastive)
        # DivergenceLoss needs the batch size on every call – we create it
        # on-the-fly in *training_step* so no need to store it here.

    # ------------------------------------------------------------------
    # Forward & helpers
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Reconstruct *x*.

        Notes
        -----
        You can call *model(x)* or *model.autoencoder(x)* interchangeably.
        """
        return self.autoencoder(x)

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def training_step(self, batch: torch.Tensor, batch_idx: int):  # type: ignore[override]
        x = batch.float()
        batch_size = x.size(0)

        # Forward pass
        embedding = self.encoder(x)
        recon_x = self.decoder(embedding)

        # ------------------------------------------------------------------
        # Loss components
        # ------------------------------------------------------------------
        recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction="mean")

        ln_loss = self.hparams.coef_ln * torch.norm(embedding, p=self.hparams.ln_parm) / batch_size
        # Keep behaviour of the original code where ln loss could be 0
        if ln_loss == 0:
            ln_loss = torch.tensor(0.5, device=self.device)

        contrastive_loss = (
            self.contrastive_loss_fn(embedding) if self.hparams.coef_contrastive else torch.tensor(0.0, device=self.device)
        )

        divergence_loss_fn = DivergenceLoss(batch_size, self.hparams.coef_divergence).to(self.device)
        divergence_loss = (
            divergence_loss_fn(embedding) if self.hparams.coef_divergence else torch.tensor(0.0, device=self.device)
        )

        # Aggregate exactly as before: mse + ln + contrastive - divergence
        loss = recon_loss + ln_loss + contrastive_loss - divergence_loss

        # Logging – *on_step* because a scheduler expects per-batch logging
        self.log_dict(
            {
                "train/loss": loss,
                "train/recon": recon_loss,
                "train/ln": ln_loss,
                "train/contrastive": contrastive_loss,
                "train/divergence": divergence_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    # You can plug a *validation_step* in the same fashion if desired

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        if self.hparams.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.hparams.learning_rate,
                max_lr=self.hparams.max_learning_rate,
                step_size_up=15,
                cycle_momentum=False,
            )
            # Lightning expects a dict to attach the cyclic scheduler per batch
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",  # update after *every* batch
                    "frequency": 1,
                },
            }
        return optimizer

    # ------------------------------------------------------------------
    # Utilities – retain convenience of the original class
    # ------------------------------------------------------------------
    def get_embedding(self, data: torch.Tensor | np.ndarray, batch_size: int = 32) -> torch.Tensor:
        """Vectorise *data* to its latent representation (no grad)."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        data = data.reshape(-1, data.shape[-2], data.shape[-1])  # ensure 3-D [N,C,H] like original
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)

        embeddings: list[torch.Tensor] = []
        self.eval()
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device).float()
                embeddings.append(self.encoder(batch))
        return torch.cat(embeddings, dim=0).cpu()

    def generate_spectra(self, embedding: torch.Tensor | np.ndarray) -> np.ndarray:
        """Decode latent vectors back to spectra (numpy).

        If *embedding* is a 1-D vector it will be broadcast to 2-D.
        """
        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding)
        embedding = torch.atleast_2d(embedding).float().to(self.device)
        self.eval()
        with torch.no_grad():
            spectra = self.decoder(embedding)
        return spectra.cpu().numpy()

# -----------------------------------------------------------------------------
# Extra: minimal *LightningDataModule* stub (optional)
# -----------------------------------------------------------------------------
class SpectraDataModule(pl.LightningDataModule):
    """Very simple wrapper to turn a *torch.Tensor* into train dataloaders."""

    def __init__(self, data: torch.Tensor, batch_size: int = 32):
        super().__init__()
        self.data = data
        self.batch_size = batch_size

    def setup(self, stage: str | None = None):  # noqa: D401, ARG002
        # Flatten to the expected 3-D shape used by the model
        self.train_data = self.data.reshape(-1, self.data.shape[-2], self.data.shape[-1])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

# -----------------------------------------------------------------------------
# End of file
# -----------------------------------------------------------------------------
