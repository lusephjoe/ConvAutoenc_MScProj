import hyperspy.api as hs
import torch

from m3_learning.nn.STEM_AE.Viz import Viz
from m3_learning.nn.STEM_AE.STEM_AE import VariationalAutoencoder
from m3_learning.viz.style import set_style
from m3_learning.viz.printing import printer
from m3_learning.nn.STEM_AE.Dataset import STEM_Dataset

import numpy as np

SEED = 42
# Set the random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# Specify the filename and the path to save the file
save_path = 'Savedata/STEM_VAE_Data'
print(f"Save path is {save_path}")
fig_path = save_path.replace("Data", "Figures") + '/'
print(f"Figures will be saved at {fig_path}")

# builds the printer object
printing = printer(basepath=fig_path, fileformats=['png', 'svg'], verbose=False)

# Set the style of the plots
set_style("printing")

vortex = STEM_Dataset(
    data_path=f"{save_path}/uncropped.hspy"
)

model = VariationalAutoencoder(
    encoder_step_size=encoder_step_size,
    pooling_list=pooling_list,
    decoder_step_size=decoder_step_size,
    upsampling_list=upsampling_list,
    embedding_size=embedding_size,
    conv_size=conv_size,
    device=device,
    learning_rate=3e-5,
)
