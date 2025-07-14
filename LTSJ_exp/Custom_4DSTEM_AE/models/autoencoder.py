"""Convolutional autoencoder for 4D‑STEM diffraction patterns."""
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, padding_mode="zeros"),  # (Q/2)
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, padding_mode="zeros"), # (Q/4)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, padding_mode="zeros"), # (Q/8)
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        # infer flattened size lazily
        self._latent = nn.Linear(64 *  (8*8), latent_dim)  # 8×8 is safe for ≤1024‑pix patterns

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self._latent(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim: int = 128, out_shape: tuple[int,int]=(128,128)):
        super().__init__()
        qy, qx = out_shape
        self._qy, self._qx = qy, qx
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.ReLU(inplace=True)
        )
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
            nn.Sigmoid()  # assumes input intensities have been normalised to [0,1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, 64, 8, 8)
        x = self.conv(x)
        # crop/pad to exact target size if needed
        return x[..., :self._qy, :self._qx]

class Autoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128, out_shape: tuple[int,int]=(128,128)):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim, out_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return latent representation without decoding."""
        return self.encoder(x)