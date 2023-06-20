from skimage.segmentation import clear_border
from skimage.morphology import closing
from skimage.measure import label
import numpy as np
from math import ceil
from openbabel import pybel
from numpy import ndarray
from utils.feature_extractor import FeatureExtractor
from typing import Union


class DensityTransformer:
    """
    Transform molecule atoms to 4D array having it's features arrays distributed in 3D space
    so that it can be used as model input. X[Y[Z[AtomsFeatures[]]]]
    """

    def __init__(
        self, max_dist: Union[float, int] = 10.0, scale: Union[float, int] = 1.0
    ) -> None:
        # validate arguments
        if scale is None:
            raise ValueError("scale must be set to make predictions")
        if not isinstance(scale, (float, int)):
            raise TypeError("scale must be number")
        if scale <= 0:
            raise ValueError("scale must be positive")

        if not isinstance(max_dist, (float, int)):
            raise TypeError("max_dist must be number")
        if max_dist <= 0:
            raise ValueError("max_dist must be positive")

        # initialize attributes
        self.max_dist = float(max_dist)
        """
        Maximum distance (in Angstroms) between atom and box center. Resulting box has
        size of 2*`max_dist`+1 Angstroms and atoms that are too far away are not included.
        """
        self.scale = float(scale)
        """Make atoms bigger (> 1) or smaller (< 1) inside the grid."""
        self.resolution = 1.0 / self.scale
        """Resolution of a grid (in Angstroms)."""
        self.box_size = ceil(2 * self.max_dist / self.resolution + 1)
        """Grid box size (in Angstroms)."""
        self.step = np.array([1.0 / self.scale] * 3)
        """Step is the dimension (in Angstroms) of one cell in the new scaled 3D grid."""
        self.fe = FeatureExtractor()
        """Feature Extractor to get atoms coordinates and features."""

        self.mol: pybel.Molecule = None
        self.coords: ndarray = None
        self.features: ndarray = None
        self.centroid: ndarray = None
        self.origin: ndarray = None

        pass

    def setMol(self, mol: pybel.Molecule):
        if not isinstance(mol, pybel.Molecule):
            raise TypeError(
                "mol should be a pybel.Molecule object, got %s " "instead" % type(mol)
            )

        self.mol = mol

        prot_coords, prot_features = self.fe.get_feature(self.mol)
        self.coords = prot_coords
        self.features = prot_features
        self.centroid = None
        self.origin = None

        return self

    def transform(self) -> ndarray:
        self._translateToCenter()
        self._scaleAndCrop()
        mol_grid = self._insertInFixed3DGrid()

        return mol_grid

    def _translateToCenter(self) -> None:
        """Move centroid to zero origin in 3D space"""
        self.centroid = self.coords.mean(axis=0)
        self.coords -= self.centroid
        self.origin = self.centroid - self.max_dist

        pass

    def _scaleAndCrop(self) -> None:
        # translate with max included distance and scale it
        grid_coords = (self.coords + self.max_dist) / self.resolution

        # converts data to nearest integers
        grid_coords = grid_coords.round().astype(int)

        # crop and return non cropped atoms coords and features only
        in_box = ((grid_coords >= 0) & (grid_coords < self.box_size)).all(axis=1)
        self.coords, self.features = grid_coords[in_box], self.features[in_box]

        pass

    def _insertInFixed3DGrid(self) -> ndarray:
        """
        Merge atom coordinates and features both represented as 2D arrays into one
        fixed-sized 3D box.

        Returns
        -------
        grid: np.ndarray, shape = (M, M, M, F)
            4D array with atom properties distributed in 3D space. M is equal to
            2 * `max_dist` / `resolution` + 1
        """

        num_features = len(self.fe.FEATURE_NAMES)
        grid_shape = (1, self.box_size, self.box_size, self.box_size, num_features)

        # init empty grid
        grid = np.zeros(
            grid_shape,
            dtype=np.float32,
        )

        # put atoms features in it's transformed coords
        for (x, y, z), f in zip(self.coords, self.features):
            grid[0, x, y, z] += f

        return grid

    def _segment_atoms(
        self, grid_probablity_map: ndarray, threshold: int = 0.5, min_size: int = 50
    ):
        """
        Extract predicted pockets from the probablity map and return a 3D grid with labeled pockets.

        Parameters
        ----------

        threshold: int
            Atoms are considered sites if thier probablity was higher than threshold.
        min_size: int
            Predicted pockets with size smaller than min_size will be excluded

        Returns
        ----------
        grid_labeled_pockets: ndarray
            3D grid with labeled predicted pockets

        pockets_num: int
            number of predicted pockets
        """

        if len(grid_probablity_map) != 1:
            raise ValueError("Segmentation of more than one molecule is not supported")

        # turn every atom to 1 (site atom) or 0 (not site atom) based on it's probability
        grid_sites = (grid_probablity_map[0] > threshold).any(axis=-1)

        # merge close site atoms
        grid_sites = closing(grid_sites)

        # exclude most site atoms that are on grid edges/border
        grid_sites = clear_border(grid_sites)

        # label every pocket of connected site atoms in grid and get how many pockets were there
        grid_labeled_pockets, pockets_num = label(grid_sites, return_num=True)

        # voxel_size represents how many atoms were squashed into one when we scaled grid
        voxel_size = (1 / self.scale) ** 3

        for pocket_label in range(1, pockets_num + 1):
            grid_pocket = grid_labeled_pockets == pocket_label

            scaled_pocket_size = grid_pocket.sum()

            # get number of atoms after reversing scale
            pocket_final_size = scaled_pocket_size * voxel_size

            if pocket_final_size < min_size:
                # pocket size is very small so exclude those atoms from being site atoms
                grid_labeled_pockets[np.where(grid_pocket)] = 0
                pockets_num -= 1

        return grid_labeled_pockets, pockets_num
