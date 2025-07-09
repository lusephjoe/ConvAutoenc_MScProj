import os
import re
import tempfile
from pathlib import PureWindowsPath

import torch
import pytest


# ────────────────────────────── helpers ────────────────────────────── #

RESERVED_CHARS_RE = re.compile(r'[<>:"/\\|?*]')  # illegal in Windows file names
MAX_FILENAME_LENGTH = 255                        # Win32 limit for a *single* name


def build_file_path(folder_path: str,
                    epoch: int,
                    coef_1: float,
                    lr: float,
                    train_loss: float) -> str:
    """
    Re-implements the exact formatting logic in the original snippet.
    """
    lr_str = format(lr, ".5f")
    return (
        f"{folder_path}/Weight_"
        f"epoch_{epoch:04d}_l1coef_{coef_1:.4f}"
        f"_lr_{lr_str}_trainloss_{train_loss:.4f}.pkl"
    )


def is_valid_windows_filename(path: str) -> bool:
    """
    True if the *basename* contains no reserved characters and isn’t too long.
    """
    filename = PureWindowsPath(path).name
    return (
        not RESERVED_CHARS_RE.search(filename) and
        len(filename) <= MAX_FILENAME_LENGTH
    )


# ────────────────────────────── tests ──────────────────────────────── #

@pytest.mark.parametrize(
    "epoch,coef_1,lr,train_loss",
    [
        (0,      0.0001, 1e-3,    0.1234),       # minimal values
        (42,     1.2345, 0.98765, 98.7654),      # typical floating-point combo
        (9999,   0.5678, 0.00001, 0.00001),      # high epoch
        (123456, 0.1,    10.0,    1e3),          # extremely long numbers
        (-1,    -0.25,   0.5,   -0.75),          # negative values still legal
    ],
)
def test_path_has_no_illegal_windows_characters(tmp_path,
                                                epoch, coef_1, lr, train_loss):
    folder_path = str(tmp_path)  # temp dir ensures write permission
    file_path = build_file_path(folder_path, epoch, coef_1, lr, train_loss)

    # 1️⃣  Pure Windows validation (even on non-Windows runners)
    assert is_valid_windows_filename(file_path), (
        f"Generated filename {file_path!r} contains illegal Windows characters "
        "or exceeds length limits."
    )

    # 2️⃣  Path length must be < 260 for legacy Win32 APIs
    assert len(str(PureWindowsPath(file_path))) < 260, (
        f"Full path exceeds 260-character Win32 limit: {len(file_path)}"
    )


@pytest.mark.parametrize(
    "epoch,coef_1,lr,train_loss",
    [
        (1, 0.001, 0.00001, 0.001),
        (200, 0.321, 0.02, 123.456),
    ],
)
def test_torch_save_does_not_raise_oserror(tmp_path,
                                           epoch, coef_1, lr, train_loss):
    folder_path = str(tmp_path)
    file_path = build_file_path(folder_path, epoch, coef_1, lr, train_loss)

    checkpoint_stub = {"dummy_tensor": torch.zeros(1)}

    # This is the critical step mirroring your snippet.
    torch.save(checkpoint_stub, file_path)

    assert os.path.isfile(file_path), "torch.save failed to create the file"


def test_filename_edge_case_near_windows_limit(tmp_path):
    """
    Build the *longest* legal filename we can with the given template
    to ensure we never exceed the Windows 255-character single-name limit.
    """
    epoch = 99999999             # 8 digits
    coef_1 = 12345.6789
    lr = 123.45678
    train_loss = 98765.4321

    file_path = build_file_path(str(tmp_path), epoch, coef_1, lr, train_loss)
    filename = PureWindowsPath(file_path).name

    assert len(filename) <= MAX_FILENAME_LENGTH, (
        f"Filename length {len(filename)} exceeds Windows limit"
    )
    assert not RESERVED_CHARS_RE.search(filename), (
        "Filename unexpectedly contains reserved characters"
    )
