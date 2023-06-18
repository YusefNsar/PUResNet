from openbabel import pybel
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import seaborn as sns

sys.path.append("/home/yusef/Development/FCI/PUResNet")

# from utils.feature_extractor import FeatureExtractor
from model.density_transformer import DensityTransformer


def make_output_folder(output_folder_path, input_folder_path):
    protein_name = os.path.basename(input_folder_path)
    o_path = os.path.join(output_folder_path, protein_name)
    # print(o_path)
    if not os.path.exists(o_path):
        os.mkdir(o_path)
    return o_path


mol = next(pybel.readfile("mol2", "test/1a2n_1/protein.mol2"))

# o_path = make_output_folder("test/output", "test/1a2n_1/protein.mol2")

# fe = FeatureExtractor()
# print(len(mol.atoms))
# print(fe.get_feature(mol)[1][0])
# print(fe.FEATURE_NAMES)

print(mol.atoms[0].coords)
ogxs = []
ogys = []
ogzs = []

for atom in mol.atoms:
    x, y, z = atom.coords
    ogxs.append(x)
    ogys.append(y)
    ogzs.append(z)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(ogxs, ogys, ogzs, s=50, alpha=0.6, edgecolors="w")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()

exit()

dt = DensityTransformer()
grid = dt.setMol(mol).transform()
# print(grid)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

xs = dt.coords[:, 0]
ys = dt.coords[:, 1]
zs = dt.coords[:, 2]
ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors="w")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
