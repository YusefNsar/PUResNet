import sys

sys.path.append("/home/yusef/Development/FCI/PUResNet")

from openbabel import pybel
import matplotlib.pyplot as plt
from utils.feature_extractor import FeatureExtractor

mol = next(pybel.readfile("mol2", "test/1a2n_1/protein.mol2"))

fe = FeatureExtractor()
coords, features = fe.get_feature(mol)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

xs = coords[:, 0]
ys = coords[:, 1]
zs = coords[:, 2]
ax.scatter(xs, ys, zs, s=50, alpha=0.6, edgecolors="w")

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()
