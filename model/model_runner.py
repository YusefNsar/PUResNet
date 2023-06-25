# from feature_extractor import FeatureExtractor
from typing import List
from openbabel import pybel
from utils.mol_3d_grid import Mol3DGrid
from model.PUResNet import PUResNet
import numpy as np


class ModelRunner:
    def __init__(self) -> None:
        self.mol_grid = Mol3DGrid(max_dist=35.0, scale=0.5)
        d = self.mol_grid.box_size
        f = len(self.mol_grid.fe.FEATURE_NAMES)

        # get box size and feature number to determine input shape to model
        self.model = PUResNet(d, f)

        pass

    def predictBindingSites(self, mol: pybel.Molecule) -> List[pybel.Molecule]:
        # save mol in grid with the required max_dist and scaling
        grid = self.mol_grid.setMol(mol).transform()

        # load trained weights
        self.model.load_weights(
            "/home/yusef/Development/FCI/PUResNet/whole_trained_model1.hdf"
        )

        # predict and extract predicted pockets
        x = self.model.predict(np.array([grid]))
        pockets = self.mol_grid.segment_grid_to_pockets(x).get_pockets_mols()

        return pockets
