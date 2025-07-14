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

def show(model: nn.Module, example_input: torch.Tensor, *_, **__):
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
