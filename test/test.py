from openbabel import pybel
import os
import sys
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns

sys.path.append("/home/yusef/Development/FCI/PUResNet")

# from utils.feature_extractor import FeatureExtractor
from utils.mol_3d_grid import Mol3DGrid
from model.PUResNet import PUResNet


mol = next(pybel.readfile("mol2", "test/1a2n_1/protein.mol2"))

# o_path = make_output_folder("test/output", "test/1a2n_1/protein.mol2")

# fe = FeatureExtractor()
# print(len(mol.atoms))
# print(fe.get_feature(mol)[1][0])
# print(fe.FEATURE_NAMES)

# mol_grid = Mol3DGrid(max_dist=35.0, scale=0.5)
# grid = mol_grid.setMol(mol).transform()

# d = mol_grid.box_size
# f = len(mol_grid.fe.FEATURE_NAMES)
# model = PUResNet(d, f)
# model.load_weights("/home/yusef/Development/FCI/PUResNet/whole_trained_model1.hdf")
# model.summary()
# x = model.predict(grid)
# print(x.sum(), x)

# with open("prediction.pickle", "wb") as f:
#     pickle.dump(x, f)

x = None
with open("prediction.pickle", "rb") as f:
    x = pickle.load(f)

mol_grid = Mol3DGrid(max_dist=35.0, scale=0.5)
grid = mol_grid.setMol(mol).transform()

pockets = mol_grid.segment_grid_to_pockets(x).get_pockets_mols()
print(pockets, len(pockets[0].atoms))
