"""Preprocess raw 4D-STEM HDF5 datasets into training tensors."""
import argparse, h5py, numpy as np, torch
from pathlib import Path

def normalise(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32")
    x -= x.min()
    x /= x.max() + 1e-6
    return x

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--downsample", type=int, default=1, help="Fourier‑bin factor")
    args = p.parse_args()

    with h5py.File(args.input, "r") as f:
        data = f["/data"][:]  # shape = (Ny, Nx, Qy, Qx)

    if args.downsample > 1:
        k = args.downsample
        data = data[..., ::k, ::k]

    data = normalise(data)

    # reshape to samples × 1 × Qy × Qx
    ny, nx, qy, qx = data.shape
    data = data.reshape(ny*nx, 1, qy, qx)

    torch.save(torch.from_numpy(data), args.output)
    print(f"Saved {data.shape[0]} samples → {args.output}")

if __name__ == "__main__":
    main()