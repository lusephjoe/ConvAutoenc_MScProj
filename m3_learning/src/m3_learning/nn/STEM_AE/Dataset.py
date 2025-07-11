import numpy as np
import hyperspy.api as hs


class STEM_Dataset:
    """Class for the STEM dataset.
    """

    def __init__(self, data_path):
        """Initialization of the class.

        Args:
            data_path (string): path where the hyperspy file is located
        """

        # loads the data
        s = hs.load(data_path,
                    reader="hspy",
                    lazy=False,
                    )

        # extracts the data
        self.data = s.data

        # sets the log data
        self.log_data = s

    def __len__(self):
        # total number of diffraction patterns = Ny × Nx
        return self.data.shape[0] * self.data.shape[1]

    def __getitem__(self, idx):
        """Return one (H, W) pattern by flat index."""
        Ny = self.data.shape[0]
        row = idx // Ny
        col = idx %  Ny
        return self.data[row, col]          # shape (512, 512)

    @property
    def log_data(self):
        return self._log_data

    @log_data.setter
    def log_data(self, log_data):
        # add 1 to avoid log(0)
        self._log_data = np.log(log_data.data + 1)
