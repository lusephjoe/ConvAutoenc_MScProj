# models/summary.py
"""
Pretty-print a PyTorch model *and* the tensor shapes that flow through it.

Example
-------
from models.summary import show
show(model, example_input=torch.randn(1, 1, 64, 64, device="cuda"))
"""
from collections import OrderedDict
from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import os

def _add_hooks(model: nn.Module, example: torch.Tensor):
    summary = OrderedDict()
    hooks = []

    def hook(module, inp, out):
        class_name = module.__class__.__name__
        key = f"{len(summary):03d}_{class_name}"
        summary[key] = {
            "in": tuple(inp[0].shape),
            "out": tuple(out.shape),
            "params": sum(p.numel() for p in module.parameters() if p.requires_grad),
            "trainable": any(p.requires_grad for p in module.parameters()),
        }

    for m in model.modules():
        # skip containers and the top-level module
        if m == model or isinstance(m, (nn.Sequential, nn.ModuleList)):
            continue
        hooks.append(m.register_forward_hook(hook))

    with torch.no_grad():
        model(example)

    for h in hooks:
        h.remove()

    return summary

def calculate_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Calculate reconstruction metrics between original and reconstructed images."""
    # Convert to numpy and handle batch dimension
    orig_np = original.detach().cpu().numpy()
    recon_np = reconstructed.detach().cpu().numpy()
    
    # Handle batch dimension - calculate metrics for each sample then average
    if orig_np.ndim == 4:  # (batch, channels, height, width)
        orig_np = orig_np.squeeze(1)  # Remove channel dimension
        recon_np = recon_np.squeeze(1)
    
    mse_values = []
    psnr_values = []
    ssim_values = []
    
    for i in range(orig_np.shape[0]):
        # MSE
        mse = np.mean((orig_np[i] - recon_np[i]) ** 2)
        mse_values.append(mse)
        
        # PSNR
        if mse > 0:
            psnr = peak_signal_noise_ratio(orig_np[i], recon_np[i], data_range=1.0)
            psnr_values.append(psnr)
        else:
            psnr_values.append(float('inf'))
        
        # SSIM
        ssim = structural_similarity(orig_np[i], recon_np[i], data_range=1.0)
        ssim_values.append(ssim)
    
    return {
        'mse': np.mean(mse_values),
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'mse_std': np.std(mse_values),
        'psnr_std': np.std(psnr_values),
        'ssim_std': np.std(ssim_values)
    }

def save_comparison_images(original: torch.Tensor, reconstructed: torch.Tensor, 
                          output_path: str, num_samples: int = 4):
    """Save comparison images showing original vs reconstructed patterns."""
    orig_np = original.detach().cpu().numpy()
    recon_np = reconstructed.detach().cpu().numpy()
    
    # Handle batch and channel dimensions
    if orig_np.ndim == 4:
        orig_np = orig_np.squeeze(1)
        recon_np = recon_np.squeeze(1)
    
    num_samples = min(num_samples, orig_np.shape[0])
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(orig_np[i], cmap='viridis')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(recon_np[i], cmap='viridis')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def show(model: nn.Module, example_input: torch.Tensor, output_dir: str = None, *_, **__):
    device = example_input.device
    model = model.to(device).eval()

    summary = _add_hooks(model, example_input)
    print("─" * 80)
    print(f"{'Layer':<35}{'Input → Output':<30}{'Params':>10}")
    print("─" * 80)
    total, trainable = 0, 0
    for k, v in summary.items():
        io = f"{v['in']} → {v['out']}"
        print(f"{k:<35}{io:<30}{v['params']:>10,}")
        total += v["params"]
        if v["trainable"]:
            trainable += v["params"]
    print("─" * 80)
    print(f"{'Total params':<65}{total:>10,}")
    print(f"{'Trainable':<65}{trainable:>10,}")
    print("─" * 80)
    
    # Add performance evaluation if output_dir is provided
    if output_dir and hasattr(model, 'forward'):
        print("\n" + "─" * 80)
        print("PERFORMANCE EVALUATION")
        print("─" * 80)
        
        with torch.no_grad():
            reconstructed = model(example_input)
            metrics = calculate_metrics(example_input, reconstructed)
            
            print(f"{'MSE':<20}{metrics['mse']:.6f} ± {metrics['mse_std']:.6f}")
            print(f"{'PSNR (dB)':<20}{metrics['psnr']:.2f} ± {metrics['psnr_std']:.2f}")
            print(f"{'SSIM':<20}{metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")
            
            # Save comparison images
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                comparison_path = os.path.join(output_dir, "reconstruction_comparison.png")
                save_comparison_images(example_input, reconstructed, comparison_path)
                print(f"{'Comparison saved to':<20}{comparison_path}")
        
        print("─" * 80)
