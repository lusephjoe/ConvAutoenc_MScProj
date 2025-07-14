"""Train the autoencoder using PyTorch Lightning."""
from __future__ import annotations
import argparse, torch, pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from models.autoencoder import Autoencoder
from models.summary import show

class LitAE(pl.LightningModule):
    def __init__(self, latent_dim: int, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.model = Autoencoder(latent_dim)
        self.loss = torch.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, = batch
        x_hat = self(x)
        loss = self.loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--latent", type=int, default=128)
    p.add_argument("--device", type=str, default="cpu", required=True)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--summary", type=bool, default=True)

    args = p.parse_args()

    pl.seed_everything(args.seed)

    data = torch.load(args.data)
    ds = TensorDataset(data)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)

    model = LitAE(args.latent, args.lr)

    if args.summary:          
                                    # grab one batch to build an example tensor
        sample = next(iter(dl))
        if isinstance(sample, (list, tuple)):
            sample = sample[0]    
        example = sample[:1].to(args.device)  
        show(model, example_input=example)
        print("wat u mean fam")

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="gpu" if args.gpus else "cpu", devices=args.gpus or 1)
    trainer.fit(model, dl)

    out_dir = args.output_dir
    out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / "ae.ckpt"
    trainer.save_checkpoint(ckpt)
    print(f"Model saved to {ckpt}")

if __name__ == "__main__":
    main()