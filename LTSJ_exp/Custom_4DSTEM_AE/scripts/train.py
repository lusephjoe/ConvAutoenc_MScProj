"""Train the autoencoder using PyTorch Lightning."""
import argparse, torch, pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from models.autoencoder import Autoencoder
from models.summary import show, calculate_metrics

class LitAE(pl.LightningModule):
    def __init__(self, latent_dim: int, lr: float, realtime_metrics: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.model = Autoencoder(latent_dim)
        self.loss = torch.nn.MSELoss()
        self.train_losses: list[float] = []
        self.validation_metrics: dict = {}
        self.realtime_metrics = realtime_metrics

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, = batch
        x_hat = self(x)
        loss = self.loss(x_hat, x)

        # Record for the post-run plot
        self.train_losses.append(loss.detach().cpu().item())

        # Calculate reconstruction metrics every N steps (if enabled)
        if self.realtime_metrics and batch_idx % 10 == 0:
            with torch.no_grad():
                metrics = calculate_metrics(x, x_hat)
                self.log("train_mse", metrics['mse'], prog_bar=True)
                self.log("train_psnr", metrics['psnr'], prog_bar=False)
                self.log("train_ssim", metrics['ssim'], prog_bar=False)

        # Send to the progress bar & logger
        self.log("train_loss", loss, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, = batch
        x_hat = self(x)
        loss = self.loss(x_hat, x)
        
        # Calculate detailed metrics for validation
        metrics = calculate_metrics(x, x_hat)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse", metrics['mse'], prog_bar=False)
        self.log("val_psnr", metrics['psnr'], prog_bar=True)
        self.log("val_ssim", metrics['ssim'], prog_bar=True)
        
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
    p.add_argument("--realtime_metrics", action="store_true", help="Enable real-time metrics calculation during training (may slow down training)")

    args = p.parse_args()

    pl.seed_everything(args.seed)

    data = torch.load(args.data)
    
    # Split data into train/validation (80/20 split)
    total_size = len(data)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # Create indices for splitting
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_data = data[train_indices]
    val_data = data[val_indices]
    
    train_ds = TensorDataset(train_data)
    val_ds = TensorDataset(val_data)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    model = LitAE(args.latent, args.lr, args.realtime_metrics)
    
    if args.summary:          
                                    # grab one batch to build an example tensor
        sample = next(iter(train_dl))
        if isinstance(sample, (list, tuple)):
            sample = sample[0]    
        example = sample[:1].to(args.device)  
        show(model, example_input=example, output_dir=args.output_dir)

        # ---------- logging ----------
    tb_logger = TensorBoardLogger(
        save_dir=args.output_dir,
        name="tb_logs",
        default_hp_metric=False,
    )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if args.gpus else "cpu",
        devices=args.gpus or 1,
        logger=tb_logger,                 
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dl, val_dl)

    # ---------- checkpoint ----------
    out_dir = args.output_dir
    out_dir.mkdir(exist_ok=True)
    ckpt = out_dir / "ae.ckpt"
    trainer.save_checkpoint(ckpt)
    print(f"Model saved to {ckpt}")

    # ---------- loss curve ----------
    loss_curve_path = args.output_dir / "loss_curve.png"
    plt.figure()
    plt.plot(model.train_losses)
    plt.xlabel("batch")
    plt.ylabel("MSE loss")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=300)
    print(f"Loss curve saved to {loss_curve_path}")
    
    # ---------- final evaluation ----------
    print("\n" + "="*80)
    print("FINAL MODEL EVALUATION")
    print("="*80)
    
    model.eval()
    with torch.no_grad():
        # Evaluate on a batch from validation set
        val_sample = next(iter(val_dl))
        if isinstance(val_sample, (list, tuple)):
            val_sample = val_sample[0]
        val_input = val_sample.to(args.device)
        val_output = model(val_input)
        
        final_metrics = calculate_metrics(val_input, val_output)
        print(f"Validation MSE:     {final_metrics['mse']:.6f} ± {final_metrics['mse_std']:.6f}")
        print(f"Validation PSNR:    {final_metrics['psnr']:.2f} ± {final_metrics['psnr_std']:.2f} dB")
        print(f"Validation SSIM:    {final_metrics['ssim']:.4f} ± {final_metrics['ssim_std']:.4f}")
        
        # Save final comparison images
        from models.summary import save_comparison_images
        final_comparison_path = args.output_dir / "final_reconstruction_comparison.png"
        save_comparison_images(val_input, val_output, final_comparison_path, num_samples=8)
        print(f"Final comparison saved to {final_comparison_path}")
    
    print("="*80)

if __name__ == "__main__":
    main()