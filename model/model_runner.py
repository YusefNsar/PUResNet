# from feature_extractor import FeatureExtractor
from typing import List
from openbabel import pybel
from model.density_transformer import DensityTransformer
from model.PUResNet import PUResNet


class ModelRunner:
    def __init__(self) -> None:
        pass

    def predictBindingSites(self, mol: pybel.Molecule) -> List[pybel.Molecule]:
        dt = DensityTransformer(max_dist=35.0, scale=0.5)

        # save mol in grid with the required max_dist and scaling
        grid = dt.setMol(mol).transform()

        # get box size and feature number to determine input shape to model
        d = dt.box_size
        f = len(dt.fe.FEATURE_NAMES)
        model = PUResNet(d, f)

        # load trained weights
        model.load_weights(
            "/home/yusef/Development/FCI/PUResNet/whole_trained_model1.hdf"
        )

        # predict and extract predicted pockets
        x = model.predict(grid)
        pockets = dt.segment_grid_to_pockets(x).get_pockets_mols()

        return pockets
